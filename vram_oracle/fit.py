"""Fit peak VRAM as a linear function of a few interpretable features.

The model is deliberately small and readable:

    peak_mib = c0
             + c1 * weights_mib          (GGUF size on disk)
             + c2 * kv_mib(ctx)          (KV cache, sliding windows accounted for)
             + c3 * n_layer              (per-layer graph and scratch buffers)
             + c4 * activation_mib       (batch * n_embd * 2 bytes, a compute-buffer proxy)
             + c5 * is_moe               (expert routing buffers)

A linear fit is the right shape here because the underlying allocator really is close to
additive: weights, cache, and buffers are separate arenas. The value of fitting rather
than computing is that c1 and c2 come out slightly off 1.0, and c0 captures the CUDA
context plus fragmentation that no formula on the internet accounts for.

Solved by ridge-regularised normal equations in pure Python. Sixty samples and six
features does not need a numerical library.
"""

import json
import pathlib
import random

from .model_info import kv_mib

MIB = 1024 * 1024
BATCH = 512  # llama.cpp default logical batch, drives the compute buffer

FEATURES = [
    "intercept",
    "weights_mib",
    "kv_mib",
    "n_layer",
    "activation_mib",
    "is_moe",
]

# Weights and cache are the physics; the rest are candidates that have to earn their place
# by improving cross-validated error on the training split.
CORE = ["intercept", "weights_mib", "kv_mib"]
OPTIONAL = ["n_layer", "activation_mib", "is_moe"]

# A gigabyte of weights occupies a gigabyte of VRAM. That coefficient is 1.0 by physics, not
# something to be learned, and fitting it freely on measurements from four models that are all
# roughly the same size produced 1.145: the regression had no leverage on the weights axis, so
# it absorbed unmodelled per-model overhead into that term. Constraining it to 1.0 and fitting
# only the terms the data actually informs is the honest use of a thin dataset.
#
# Implemented by subtracting the known weights contribution from the target and fitting the
# remainder, which is exactly equivalent to a constrained least squares with that coefficient
# pinned, and much simpler than adding constraint machinery to the solver.
CONSTRAINED = {"weights_mib": 1.0}

HOLDOUT_SEED = 20260725
HOLDOUT_FRACTION = 0.2
MIN_HOLDOUT = 8


def featurize(info, ctx):
    return {
        "intercept": 1.0,
        "weights_mib": info["weights_mib"],
        "kv_mib": kv_mib(info, ctx),
        "n_layer": float(info["n_layer"]),
        "activation_mib": BATCH * info["n_embd"] * 2 / MIB,
        "is_moe": float(info["is_moe"]),
    }


def row(info, ctx, features=None):
    f = featurize(info, ctx)
    return [f[k] for k in (features or FEATURES)]


def solve(X, y, ridge=1e-6, names=None):
    """Ridge-regularised least squares via normal equations and Gaussian elimination."""
    names = names or FEATURES
    p = len(X[0])
    ata = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(p)] for i in range(p)]
    atb = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(p)]
    scale = max(max(abs(v) for v in r) for r in ata) or 1.0
    for i in range(p):
        ata[i][i] += ridge * scale

    aug = [ata[i] + [atb[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError(f"singular design matrix at column {col} ({names[col]})")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(p):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][p] for i in range(p)]


def predict(coef, info, ctx, features=None):
    return sum(c * v for c, v in zip(coef, row(info, ctx, features)))


def r_squared(y, yhat):
    mean = sum(y) / len(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, yhat))
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def errors(y, yhat):
    res = [a - b for a, b in zip(yhat, y)]
    n = len(res)
    return {
        "n": n,
        "mae_mib": sum(abs(r) for r in res) / n,
        "rmse_mib": (sum(r * r for r in res) / n) ** 0.5,
        "max_abs_mib": max(abs(r) for r in res),
        "bias_mib": sum(res) / n,
    }


def load_samples(path, catalog):
    """Usable measurements only: no contamination, no error, fully GPU resident.

    Partially offloaded runs are kept out of the fit. When a model does not fit, the peak
    reflects what Ollama chose to place on the GPU rather than what the configuration
    wanted, so those rows would teach the regression the wrong thing. They are returned
    separately and used to check the fit's fit/no-fit verdicts.
    """
    usable, offloaded = [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or r.get("contaminated") or r["model"] not in catalog:
            continue
        if not r.get("effective_ctx"):
            continue
        (usable if r["fully_resident"] else offloaded).append(r)
    return usable, offloaded


def split(samples):
    """Deterministic holdout, stratified so no model contributes more than its share."""
    idx = list(range(len(samples)))
    n_hold = max(MIN_HOLDOUT, int(round(len(samples) * HOLDOUT_FRACTION)))
    rng = random.Random(HOLDOUT_SEED)
    by_model = {}
    for i in idx:
        by_model.setdefault(samples[i]["model"], []).append(i)
    hold = []
    order = sorted(by_model)
    rng.shuffle(order)
    while len(hold) < n_hold:
        progressed = False
        for m in order:
            pool = [i for i in by_model[m] if i not in hold]
            if len(by_model[m]) - len([i for i in hold if samples[i]["model"] == m]) <= 1:
                continue  # never strip a model down to a single training point
            if pool and len(hold) < n_hold:
                hold.append(rng.choice(pool))
                progressed = True
        if not progressed:
            break
    hold = sorted(hold)
    train = [i for i in idx if i not in hold]
    return train, hold


def fit(samples, catalog, train_idx=None, features=None, constrain=True):
    """Least squares over `features`.

    With `constrain`, any feature in CONSTRAINED is held at its physical value rather than
    estimated: its contribution is subtracted from the target and the remaining coefficients
    are fitted against the residual. The returned coefficient vector still carries the pinned
    value in its slot, so callers and the report see a full coefficient set either way.
    """
    features = features or FEATURES
    train_idx = list(range(len(samples))) if train_idx is None else train_idx
    pinned = {k: v for k, v in CONSTRAINED.items() if constrain and k in features}
    free = [f for f in features if f not in pinned]

    rows, targets = [], []
    for i in train_idx:
        info = catalog[samples[i]["model"]]
        f = featurize(info, samples[i]["effective_ctx"])
        offset = sum(coef * f[name] for name, coef in pinned.items())
        rows.append([f[k] for k in free])
        targets.append(float(samples[i]["delta_mib"]) - offset)

    if not free:
        return [pinned.get(k, 0.0) for k in features]
    solved = solve(rows, targets, names=free)
    lookup = dict(zip(free, solved))
    lookup.update(pinned)
    return [lookup[k] for k in features]


def loo_cv_mae(samples, catalog, idx, features):
    """Leave-one-out CV error over `idx`. Used for feature selection, never for reporting."""
    total = 0.0
    for held in idx:
        rest = [i for i in idx if i != held]
        try:
            coef = fit(samples, catalog, rest, features)
        except ValueError:
            return float("inf")
        s = samples[held]
        pred = predict(coef, catalog[s["model"]], s["effective_ctx"], features)
        total += abs(pred - s["delta_mib"])
    return total / len(idx)


def select_features(samples, catalog, train_idx):
    """Pick the feature set with the best leave-one-out error on the training split.

    Selection touches only training data, so the holdout numbers reported afterwards stay
    an honest estimate of error on configurations the fit has never seen.
    """
    results = []
    for mask in range(1 << len(OPTIONAL)):
        feats = CORE + [f for k, f in enumerate(OPTIONAL) if mask >> k & 1]
        results.append((loo_cv_mae(samples, catalog, train_idx, feats), feats))
    results.sort(key=lambda r: (r[0], len(r[1])))
    return results[0][1], [
        {"features": f, "loo_mae_mib": round(m, 1)} for m, f in sorted(results, key=lambda r: r[0])
    ]


def evaluate(coef, samples, catalog, idx, features=None):
    y = [float(samples[i]["delta_mib"]) for i in idx]
    yhat = [predict(coef, catalog[samples[i]["model"]], samples[i]["effective_ctx"], features)
            for i in idx]
    out = errors(y, yhat)
    out["r2"] = r_squared(y, yhat) if len(y) > 2 else float("nan")
    out["worst"] = sorted(
        (
            {
                "model": samples[i]["model"],
                "ctx": samples[i]["effective_ctx"],
                "actual_mib": round(y[k]),
                "predicted_mib": round(yhat[k]),
                "error_mib": round(yhat[k] - y[k]),
            }
            for k, i in enumerate(idx)
        ),
        key=lambda d: -abs(d["error_mib"]),
    )[:6]
    return out
