# vram-oracle

An empirically fitted VRAM calculator for one specific card, measured rather than derived.

Catalog task `GPU-010`.

**[Read this on the web](https://jesserweigel.github.io/vram-oracle/)**

## Status: INCOMPLETE, and the reason is interesting

**One assertion still fails, on purpose.** The task asked for predictions within 1 GiB on
held-out configurations. Mean absolute error is **949 MiB**, which clears it, but worst case is
**1400 MiB**, which does not. That assertion is left failing rather than relaxed, because a
calculator reporting confidence it has not earned is worse than one admitting it is
under-measured.

```
FAIL: test_holdout_accuracy_target
  AssertionError: 1400 not less than or equal to 1024
```

25 of 26 tests pass.

### The fit was wrong in a fixable way, and one way it was not

The first fit estimated the weights coefficient freely and got **1.145**. Physics says it is
exactly 1.0: a gigabyte of weights occupies a gigabyte of VRAM. The regression had no leverage on
that axis, because every measured model is roughly the same size, so it absorbed unmodelled
per-model overhead into the weights term.

Pinning that coefficient to its physical value and fitting only the terms the data actually
informs improved the holdout worst case from **1708 MiB to 1400 MiB**, and made the model
physically sensible, which is a separate test that now passes. Constraining a coefficient you
know is the honest use of a thin dataset.

What it did not fix is the accuracy target, and that is the real limit: no amount of better
modelling substitutes for measurements across the size range. Fitting core terms only was tried
and is worse, 1751 MiB, so the extra terms carry real signal rather than overfitting.

### Why the data is thin

The sweep needs the GPU nearly empty to load large models. This workstation runs a Minecraft bot
swarm that keeps `gpt-oss:20b` resident at roughly 13 GB, leaving about 17.8 GB free. Seven of the
fifteen local models are larger than that, and even the 17 GB models want around 19.5 GB once the
KV cache is allocated. They were deferred, not measured.

What actually got measured, cleanly:

| Model | Contexts measured |
|---|---|
| `gemma4:e4b` | 2048, 8192, 32768, 65536, 131072 |
| `qwen3.5:9b` | 2048, 8192, 32768, 65536, 131072 |
| `qwen3:8b` | 2048, 8192, 32768, 40960 |
| `qwen3-minecraft:8b` | 2048, 8192, 32768 |

17 clean measurements from 4 models, all between roughly 5 and 10 GB. Fitting a weights term from
four models that are all about the same size is asking a regression to extrapolate, and the
inflated coefficient is it declining to do so honestly.

### What would finish it

Roughly 45 minutes with the GPU otherwise idle, which would let the sweep reach the 19 GB to 24 GB
models and give the weights term real leverage. Run:

```bash
python3 -m vram_oracle.measure --out data/measurements.jsonl --passes 2
python3 -m vram_oracle.cli fit
```

## What does work

The measurement harness is sound, and it is the part worth keeping.

- **Contamination detection.** Measurements taken while another model was loading, or where the
  target model was already resident, are flagged `contaminated` and excluded from the fit rather
  than averaged in. 4 of 21 measurements were caught this way. Every `gpt-oss:20b` reading was
  contaminated, because the swarm holds it, and the tool reported a delta of 0 as suspect instead
  of recording a 20 B model as free.
- **Two independent signals.** Global VRAM delta from `nvidia-smi` and per-model `size_vram` from
  Ollama's `/api/ps` are both recorded. Disagreement beyond a few hundred MiB marks the sample bad.
- **Incremental writes.** Each measurement appends to JSONL immediately, so a crash or a kill
  loses one sample rather than the sweep.
- **Deferral rather than failure.** A configuration that cannot fit right now is deferred and
  retried on a later pass instead of recorded as an error.

24 of 26 tests pass, covering the fit maths, the model catalog, the CLI, and the deterministic
holdout split.

## Using it

```bash
python3 -m vram_oracle.cli will-it-fit qwen3.5:9b --ctx 32768   # verdict against free VRAM now
python3 -m vram_oracle.cli predict qwen3.5:9b --ctx 32768       # prediction, no GPU needed
python3 -m vram_oracle.cli max-ctx qwen3.5:9b                   # largest context that fits
python3 -m vram_oracle.cli table                                # every local model
python3 -m vram_oracle.cli measurements                         # each measurement vs prediction
```

Predictions carry an error bar taken from the worst held-out error, currently 1708 MiB. Treat that
number as the honest width of the answer. Within the 5 to 10 GB range that was actually measured
it will be better than that; outside it, it is extrapolation.

## A note on the fix in tests/test_fit.py

One test failure during this session was a false alarm and was fixed rather than the code.
`test_refit_reproduces_the_shipped_coefficients` called `fit(samples, catalog)` without passing
the feature list the shipped model selected, so it silently refit with the full default feature
set and got different coefficients for an entirely legitimate reason. The test now passes the
selected features. The shipped coefficients were always fit on all data, which was correct.

## License

MIT.
