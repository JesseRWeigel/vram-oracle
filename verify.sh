#!/usr/bin/env bash
# Verify command for vram-oracle. Exit 0 means the calculator works.
#
#   ./verify.sh          unit tests, refit determinism, CLI behaviour, holdout accuracy
#   ./verify.sh --live   the above, plus a real measurement on the GPU compared to the
#                        prediction. Needs Ollama running and free VRAM.

set -euo pipefail
cd "$(dirname "$0")"

echo "== unit tests"
python3 -m unittest discover -s tests -q

echo
echo "== refit from committed measurements reproduces the shipped model"
python3 - <<'PY'
import json, pathlib, sys
from vram_oracle import fit as f
m = json.loads(pathlib.Path("model.json").read_text())
samples, offloaded = f.load_samples("data/measurements.jsonl", m["catalog"])
assert len(samples) == m["n_measurements"], (len(samples), m["n_measurements"])
train, hold = f.split(samples)
coef = f.fit(samples, m["catalog"], train)
stats = f.evaluate(coef, samples, m["catalog"], hold)
for k, v in zip(f.FEATURES, coef):
    assert abs(v - m["coefficients_train_only"][k]) < 1e-6, k
print(f"  {len(samples)} measurements, {len(hold)} held out, {len(offloaded)} offloaded rows excluded")
print(f"  holdout MAE {stats['mae_mib']:.0f} MiB, worst {stats['max_abs_mib']:.0f} MiB, R2 {stats['r2']:.4f}")
assert stats["max_abs_mib"] <= 1024, f"holdout worst error {stats['max_abs_mib']:.0f} MiB exceeds the 1 GiB target"
PY

echo
echo "== CLI answers a question"
python3 -m vram_oracle.cli will-it-fit qwen3:32b --ctx 32768 --free-mib 31000
python3 -m vram_oracle.cli max-ctx qwen3.6:35b-a3b --free-mib 31000

if [ "${1:-}" = "--live" ]; then
  echo
  echo "== live check: measure a real configuration and compare to the prediction"
  python3 - <<'PY'
import json, pathlib, sys
from vram_oracle import cli, measure, model_info
m = json.loads(pathlib.Path("model.json").read_text())
name, ctx = "qwen3:8b", 16384
info = cli.resolve_info(name, m)
pred = cli.predict_mib(m, info, ctx)
need = measure.room_needed(info, ctx)
base, foreign = measure.wait_for_room(need, timeout=600)
if base is None:
    sys.exit("BLOCKED: no free VRAM for the live check within 10 minutes")
rec = measure.measure(name, ctx, base, foreign)
measure.unload(name)
err = rec["delta_mib"] - pred
print(f"  predicted {pred:,.0f} MiB, measured {rec['delta_mib']:,} MiB, error {err:,.0f} MiB")
assert not rec["contaminated"], "another process changed GPU allocation mid-measurement"
assert abs(err) <= m["error_bar_mib"], f"live error {err:.0f} exceeds the {m['error_bar_mib']} MiB bar"
PY
fi

echo
echo "VERIFY OK"
