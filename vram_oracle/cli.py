"""vram-oracle: will this model at this context fit on this card?

Answers come from a regression fitted to measurements taken on the machine it runs on,
not from a formula. `vram-oracle fit` regenerates the coefficients from
data/measurements.jsonl; the shipped model.json is the fit from the sweep in the README.
"""

import argparse
import json
import pathlib
import sys

from . import fit as fitmod
from . import model_info

MODEL_JSON = pathlib.Path(__file__).resolve().parent.parent / "model.json"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "measurements.jsonl"


def load_model(path=None):
    p = pathlib.Path(path) if path else MODEL_JSON
    if not p.exists():
        sys.exit(f"no fitted model at {p}. Run: vram-oracle fit")
    return json.loads(p.read_text())


def gpu_state():
    from .measure import gpu_total_mib, gpu_used_mib

    try:
        total, used = gpu_total_mib(), gpu_used_mib()
        return {"total_mib": total, "used_mib": used, "free_mib": total - used}
    except Exception as e:
        return {"error": str(e)}


def resolve_info(name, m):
    """Model metadata, preferring the live Ollama daemon and falling back to the cache."""
    cached = m.get("catalog", {})
    try:
        info = model_info.describe(name)
        disk = next((x["disk_bytes"] for x in model_info.list_models() if x["name"] == name), 0)
        if disk:
            info["disk_bytes"] = disk
            info["weights_mib"] = disk / fitmod.MIB
            return info
    except Exception:
        pass
    if name in cached:
        return cached[name]
    matches = [k for k in cached if k.split(":")[0] == name.split(":")[0]]
    if len(matches) == 1:
        return cached[matches[0]]
    sys.exit(
        f"unknown model {name!r}: Ollama is not reachable and it is not in the cached "
        f"catalog. Known: {', '.join(sorted(cached)) or 'none'}"
    )


def predict_mib(m, info, ctx):
    coef = [m["coefficients"][f] for f in m["features"]]
    return fitmod.predict(coef, info, ctx, m["features"])


def breakdown(m, info, ctx):
    f = fitmod.featurize(info, ctx)
    return [(k, f[k], m["coefficients"][k], f[k] * m["coefficients"][k]) for k in m["features"]]


def cmd_predict(a):
    m = load_model(a.model_json)
    info = resolve_info(a.model, m)
    pred = predict_mib(m, info, a.ctx)
    err = m["error_bar_mib"]
    if a.json:
        print(json.dumps({
            "model": info["name"], "ctx": a.ctx,
            "predicted_mib": round(pred), "error_bar_mib": round(err),
        }))
        return 0
    print(f"{info['name']} @ ctx {a.ctx}: {pred:,.0f} MiB  (+/- {err:,.0f} MiB)")
    if a.explain:
        print_breakdown(m, info, a.ctx)
    return 0


def print_breakdown(m, info, ctx):
    print("\n  term              feature value    coefficient       MiB")
    for k, v, c, contrib in breakdown(m, info, ctx):
        print(f"  {k:16s} {v:14,.2f} {c:14,.4f} {contrib:9,.0f}")
    print(f"  {'arch':16s} {info['arch']} "
          f"{'MoE' if info['is_moe'] else 'dense'}, {info['n_attn_layer']} attention layers, "
          f"{info['kv_bytes_per_token_global'] / 1024:.0f} KiB/token global cache"
          + (f", {info['kv_bytes_per_token_swa'] / 1024:.0f} KiB/token windowed to "
             f"{info['sliding_window']}" if info["kv_bytes_per_token_swa"] else ""))


def cmd_fit(a):
    catalog = model_info.catalog() if a.refresh_catalog else None
    if catalog is None:
        try:
            catalog = model_info.catalog()
        except Exception:
            catalog = load_model(a.model_json).get("catalog", {})
    samples, offloaded = fitmod.load_samples(a.data, catalog)
    if len(samples) < 12:
        sys.exit(f"only {len(samples)} usable measurements in {a.data}; measure more first")

    train, hold = fitmod.split(samples)
    if a.features:
        features, selection = [f.strip() for f in a.features.split(",")], []
    else:
        features, selection = fitmod.select_features(samples, catalog, train)
    coef_train = fitmod.fit(samples, catalog, train, features)
    train_stats = fitmod.evaluate(coef_train, samples, catalog, train, features)
    hold_stats = fitmod.evaluate(coef_train, samples, catalog, hold, features)
    coef_all = fitmod.fit(samples, catalog, None, features)
    all_stats = fitmod.evaluate(coef_all, samples, catalog, list(range(len(samples))), features)

    out = {
        "features": features,
        "feature_selection_loo_on_train": selection,
        "coefficients": dict(zip(features, coef_all)),
        "coefficients_train_only": dict(zip(features, coef_train)),
        "error_bar_mib": round(hold_stats["max_abs_mib"]),
        "error_bar_basis": "worst absolute error on the held-out configurations",
        "n_measurements": len(samples),
        "n_offloaded_excluded": len(offloaded),
        "holdout": {
            "n": len(hold),
            "configs": [
                {"model": samples[i]["model"], "ctx": samples[i]["effective_ctx"]} for i in hold
            ],
            **{k: round(v, 1) for k, v in hold_stats.items() if isinstance(v, float)},
            "worst": hold_stats["worst"],
        },
        "train": {k: round(v, 1) for k, v in train_stats.items() if isinstance(v, float)},
        "all_data": {k: round(v, 1) for k, v in all_stats.items() if isinstance(v, float)},
        "worst_overall": all_stats["worst"],
        "gpu_total_mib": max((s.get("gpu_total_mib") or 0) for s in samples) or None,
        "catalog": catalog,
    }
    dest = pathlib.Path(a.out or MODEL_JSON)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"fitted on {len(train)} configurations, held out {len(hold)}")
    print(f"features chosen by leave-one-out CV on the training split: {', '.join(features)}")
    for k, v in zip(features, coef_all):
        print(f"  {k:16s} {v:12,.4f}")
    print(f"train    R2={train_stats['r2']:.4f} MAE={train_stats['mae_mib']:.0f} MiB "
          f"max={train_stats['max_abs_mib']:.0f} MiB")
    print(f"holdout  R2={hold_stats['r2']:.4f} MAE={hold_stats['mae_mib']:.0f} MiB "
          f"max={hold_stats['max_abs_mib']:.0f} MiB  (n={len(hold)})")
    print(f"wrote {dest}")
    return 0


def cmd_fits(a):
    m = load_model(a.model_json)
    info = resolve_info(a.model, m)
    pred = predict_mib(m, info, a.ctx)
    err = m["error_bar_mib"]
    worst = pred + err

    gpu = gpu_state()
    if a.free_mib is not None:
        free, source = a.free_mib, "supplied"
    elif "error" in gpu:
        sys.exit(f"cannot read the GPU ({gpu['error']}). Pass --free-mib to predict anyway.")
    else:
        free, source = gpu["free_mib"], "free right now"

    if worst <= free:
        verdict, code = "FITS", 0
    elif pred <= free:
        verdict, code = "TIGHT", 1
    else:
        verdict, code = "WILL NOT FIT", 2

    if a.json:
        print(json.dumps({
            "model": info["name"], "ctx": a.ctx, "verdict": verdict,
            "predicted_mib": round(pred), "error_bar_mib": round(err),
            "worst_case_mib": round(worst), "free_mib": free,
            "headroom_mib": round(free - pred), "gpu": gpu,
        }))
        return code

    print(f"{verdict}: {info['name']} at ctx {a.ctx}")
    print(f"  predicted peak   {pred:>9,.0f} MiB  (+/- {err:,.0f}, worst case {worst:,.0f})")
    print(f"  VRAM {source:<12} {free:>9,.0f} MiB" + (
        f"  of {gpu['total_mib']:,} total, {gpu['used_mib']:,} in use"
        if "error" not in gpu and a.free_mib is None else ""))
    print(f"  headroom         {free - pred:>9,.0f} MiB")
    if verdict == "TIGHT":
        print("  Predicted to fit, but not by more than the calculator's own error bar.")
    if a.explain:
        print_breakdown(m, info, a.ctx)
    return code


def cmd_max_ctx(a):
    """Largest context that still fits, found by bisection on the fitted prediction."""
    m = load_model(a.model_json)
    info = resolve_info(a.model, m)
    budget = a.free_mib if a.free_mib is not None else gpu_state().get("free_mib")
    if budget is None:
        sys.exit("cannot read the GPU; pass --free-mib")
    err = m["error_bar_mib"]
    lo, hi = 0, max(info["train_ctx"] or 131072, 131072)
    if predict_mib(m, info, lo) + err > budget:
        print(f"{info['name']}: does not fit at any context in {budget:,} MiB free")
        return 2
    while hi - lo > 256:
        mid = (lo + hi) // 2
        if predict_mib(m, info, mid) + err <= budget:
            lo = mid
        else:
            hi = mid
    print(f"{info['name']}: up to ctx {lo:,} fits in {budget:,} MiB free "
          f"(predicted {predict_mib(m, info, lo):,.0f} MiB, error bar {err:,.0f})")
    return 0


def cmd_table(a):
    m = load_model(a.model_json)
    catalog = m["catalog"]
    contexts = [int(c) for c in a.contexts.split(",")]
    free = a.free_mib if a.free_mib is not None else gpu_state().get("free_mib")
    print(f"predicted peak VRAM in MiB, error bar +/-{m['error_bar_mib']:,}"
          + (f", {free:,} MiB free now" if free else ""))
    print(f"{'model':28s}" + "".join(f"{c:>10,}" for c in contexts))
    for name in sorted(catalog, key=lambda n: catalog[n]["weights_mib"]):
        info = catalog[name]
        cells = []
        for c in contexts:
            v = predict_mib(m, info, c)
            mark = "" if free is None else (" " if v + m["error_bar_mib"] <= free else "!")
            cells.append(f"{v:>9,.0f}{mark}")
        print(f"{name:28s}" + "".join(cells))
    print("  ! means the worst case exceeds what is free right now")
    return 0


def cmd_measurements(a):
    m = load_model(a.model_json)
    samples, offloaded = fitmod.load_samples(a.data, m["catalog"])
    print(f"{len(samples)} usable, {len(offloaded)} excluded for partial CPU offload")
    print(f"{'model':28s}{'ctx':>8}{'actual':>9}{'predicted':>11}{'error':>8}")
    for s in sorted(samples, key=lambda s: (s["model"], s["effective_ctx"])):
        info = m["catalog"][s["model"]]
        p = predict_mib(m, info, s["effective_ctx"])
        print(f"{s['model']:28s}{s['effective_ctx']:>8,}{s['delta_mib']:>9,}"
              f"{p:>11,.0f}{p - s['delta_mib']:>8,.0f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vram-oracle", description=__doc__.splitlines()[0])
    ap.add_argument("--model-json", default=None, help="path to a fitted model.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, ctx=True):
        p.add_argument("model", help="ollama model name, e.g. qwen3:32b")
        if ctx:
            p.add_argument("--ctx", type=int, default=8192, help="num_ctx (default 8192)")
        p.add_argument("--free-mib", type=int, default=None,
                       help="assume this much free VRAM instead of reading the GPU")
        p.add_argument("--explain", action="store_true", help="show the term breakdown")
        p.add_argument("--json", action="store_true")

    common(sub.add_parser("will-it-fit", help="verdict against VRAM free right now"))
    common(sub.add_parser("predict", help="predicted peak VRAM, no GPU needed"))
    common(sub.add_parser("max-ctx", help="largest context that fits"), ctx=False)

    p = sub.add_parser("fit", help="refit from measurements")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--out", default=None)
    p.add_argument("--refresh-catalog", action="store_true")
    p.add_argument("--features", default=None,
                   help="comma separated feature list, skipping automatic selection")

    p = sub.add_parser("table", help="predictions for every local model")
    p.add_argument("--contexts", default="2048,8192,32768,65536,131072")
    p.add_argument("--free-mib", type=int, default=None)

    p = sub.add_parser("measurements", help="every measurement against its prediction")
    p.add_argument("--data", default=str(DATA))

    a = ap.parse_args(argv)
    return {
        "will-it-fit": cmd_fits,
        "predict": cmd_predict,
        "max-ctx": cmd_max_ctx,
        "fit": cmd_fit,
        "table": cmd_table,
        "measurements": cmd_measurements,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
