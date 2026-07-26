"""Measure real peak VRAM for one (model, num_ctx) configuration on the local GPU.

The measurement is a delta, not an absolute. Other processes on this box use the GPU, so
the number that matters is how much VRAM appeared between "nothing of ours loaded" and
"model loaded and generating". Any sample where the set of foreign GPU consumers changed
mid-measurement is discarded, because the delta would be attributing someone else's
allocation to our model.

Results are appended to the output file one JSON object per line, flushed after every
sample, so a crash or a kill loses at most the run in flight.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error

from . import model_info

MIB = 1024 * 1024
IDLE_CEILING_MIB = 2000  # desktop compositor and friends; above this the GPU is busy


def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return int(out[0])


def gpu_total_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return int(out[0])


def loaded_models():
    """{name: {size, size_vram, context}} for everything Ollama currently holds."""
    try:
        ps = model_info._get("/api/ps").get("models", []) or []
    except Exception:
        return {}
    return {
        m["name"]: {
            "size": m.get("size", 0),
            "size_vram": m.get("size_vram", 0),
            "context": m.get("context_length") or m.get("context") or 0,
        }
        for m in ps
    }


def unload(name):
    try:
        model_info._post("/api/generate", {"model": name, "keep_alive": 0, "prompt": ""}, timeout=120)
    except Exception:
        pass


def wait_for_room(needed_mib, timeout=900, poll=5.0, settle=2):
    """Block until there is room for `needed_mib` and the GPU stops moving.

    This box runs a fleet of agents that load their own models, so waiting for a
    completely idle GPU can mean waiting forever. What the measurement actually requires
    is weaker: enough free VRAM that our model lands fully on the GPU, and a foreign
    allocation set that holds still for the duration so the delta is ours alone. Foreign
    models are waited out rather than stopped, since another agent may be mid-generation.

    Returns (baseline_mib, foreign_set) or (None, None) on timeout.
    """
    deadline = time.time() + timeout
    total = gpu_total_mib()
    stable = 0
    last = None
    while time.time() < deadline:
        foreign = frozenset(loaded_models())
        used = gpu_used_mib()
        if last == (foreign, used // 64) and total - used >= needed_mib:
            stable += 1
            if stable >= settle:
                return used, foreign
        else:
            stable = 0
        last = (foreign, used // 64)
        time.sleep(poll)
    return None, None


class Sampler(threading.Thread):
    """Poll nvidia-smi at ~8 Hz and keep the peak."""

    def __init__(self, interval=0.12):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.n = 0
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                v = gpu_used_mib()
            except Exception:
                v = 0
            self.peak = max(self.peak, v)
            self.n += 1
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()
        self.join(timeout=5)
        return self.peak


PROMPT = "Count from one to five."


def measure(name, ctx, baseline, foreign_before=frozenset(), timeout=420):
    """Load `name` at num_ctx=`ctx`, generate a few tokens, return the peak delta."""
    foreign_before = set(foreign_before) | set(loaded_models())
    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    err = None
    try:
        model_info._post(
            "/api/generate",
            {
                "model": name,
                "prompt": PROMPT,
                "stream": False,
                "keep_alive": "60s",
                "options": {"num_ctx": ctx, "num_predict": 8, "temperature": 0},
            },
            timeout=timeout,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    ps = loaded_models()
    peak = sampler.stop()
    resident = gpu_used_mib()

    mine = ps.get(name, {})
    foreign_after = set(ps) - {name}
    contaminated = foreign_after != set(foreign_before)

    return {
        "model": name,
        "requested_ctx": ctx,
        "effective_ctx": mine.get("context", 0),
        "baseline_mib": baseline,
        "foreign_mib": baseline,
        "gpu_total_mib": gpu_total_mib(),
        "peak_mib": peak,
        "resident_mib": resident,
        "delta_mib": peak - baseline,
        "resident_delta_mib": resident - baseline,
        "ps_size_mib": mine.get("size", 0) / MIB,
        "ps_size_vram_mib": mine.get("size_vram", 0) / MIB,
        "fully_resident": bool(mine) and mine.get("size_vram", 0) >= mine.get("size", 1),
        "load_and_gen_s": round(elapsed, 1),
        "samples": sampler.n,
        "contaminated": contaminated,
        "error": err,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def room_needed(info, ctx, margin=1500):
    """Rough gate before attempting a load. Metadata only, no fitted model required."""
    return info["weights_mib"] + model_info.kv_mib(info, ctx) + margin


def run_sweep(models, contexts, out_path, idle_timeout=900, passes=3):
    """Measure every (model, context) pair, retrying the ones the fleet interfered with.

    Contaminated and skipped configurations come back around on the next pass, because a
    busy GPU is a transient condition on this box and not a property of the config.
    """
    out_path = pathlib.Path(out_path)
    catalog = model_info.catalog()
    total_vram = gpu_total_mib()

    def already_good():
        good = set()
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("contaminated") and not r.get("error") and r.get("effective_ctx"):
                    good.add((r["model"], r["requested_ctx"]))
        return good

    todo = [(m, c) for m in models for c in contexts]
    for p in range(1, passes + 1):
        done = already_good()
        pending = [t for t in todo if t not in done]
        if not pending:
            break
        # Cheapest first: small configurations find a gap in fleet GPU traffic sooner,
        # and if the sweep is cut short the surviving data still spans every model.
        pending.sort(key=lambda t: room_needed(catalog[t[0]], t[1]))
        # Early passes are impatient so the cheap configurations all land quickly; later
        # passes wait longer for the fleet to release enough VRAM for the expensive ones.
        wait = max(90, idle_timeout * p // passes)
        print(f"=== pass {p}: {len(pending)} of {len(todo)} configurations remaining, "
              f"waiting up to {wait}s for room each", flush=True)
        with out_path.open("a") as fh:
            for i, (name, ctx) in enumerate(pending, 1):
                info = catalog[name]
                need = min(room_needed(info, ctx), total_vram - 600)
                baseline, foreign = wait_for_room(need, timeout=wait)
                if baseline is None:
                    print(f"[{i}/{len(pending)}] no room for {name} ctx={ctx} "
                          f"(needs ~{need:.0f} MiB), deferring", flush=True)
                    continue
                print(f"[{i}/{len(pending)}] {name} ctx={ctx} baseline={baseline}",
                      end=" ", flush=True)
                rec = measure(name, ctx, baseline, foreign)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"-> peak={rec['peak_mib']} delta={rec['delta_mib']} "
                    f"eff_ctx={rec['effective_ctx']} vram={rec['ps_size_vram_mib']:.0f} "
                    f"{'CONTAM ' if rec['contaminated'] else ''}"
                    f"{'PARTIAL ' if not rec['fully_resident'] else ''}{rec['error'] or ''} "
                    f"({rec['load_and_gen_s']}s)",
                    flush=True,
                )
                unload(name)
                time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description="sweep peak VRAM over models and contexts")
    ap.add_argument("--out", default="data/measurements.jsonl")
    ap.add_argument("--contexts", default="2048,8192,32768,65536,131072")
    ap.add_argument("--models", default="", help="comma separated; default every local model")
    ap.add_argument("--idle-timeout", type=int, default=900)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()

    contexts = [int(c) for c in a.contexts.split(",") if c]
    models = (
        [m.strip() for m in a.models.split(",") if m.strip()]
        or [m["name"] for m in model_info.list_models()]
    )
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"gpu total {gpu_total_mib()} MiB, {len(models)} models x {len(contexts)} contexts")
    run_sweep(models, contexts, a.out, a.idle_timeout, a.passes)


if __name__ == "__main__":
    sys.exit(main())
