"""Model metadata: what the GGUF header and tensor table say about a model.

Everything here is derived from Ollama's /api/show and /api/tags. No measurement, no fitting.

The interesting part is `kv_layout`. The textbook KV formula
(2 * n_layer * n_kv_head * head_dim * 2 bytes) is wrong for most of the models on this
machine, because it assumes every layer is a full-attention layer with a global cache:

  * qwen3.5 / nemotron are hybrid stacks. Most layers are SSM layers with a fixed-size
    recurrent state, and only every Nth layer keeps a KV cache.
  * gemma4 puts most layers on sliding-window attention, so those layers cache at most
    `sliding_window` tokens no matter how large the context is.
  * Vision towers carry attention tensors that never get a text KV cache at all.

Reading the tensor table sidesteps all of that. A layer has a KV cache if and only if it
has a `blk.N.attn_k.weight` tensor, and that tensor's output dimension is exactly the
per-token K size for the layer. Layers are then bucketed by shape, and the bucket whose
head dimension matches `key_length_swa` is the sliding-window bucket.
"""

import collections
import json
import re
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
MIB = 1024 * 1024


def _post(path, payload, timeout=300):
    req = urllib.request.Request(
        OLLAMA + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(path, timeout=30):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.load(r)


def list_models():
    """Local models only. Cloud-hosted entries have no size and cannot be measured."""
    out = []
    for m in _get("/api/tags").get("models", []):
        if m.get("name", "").endswith(":cloud") or not m.get("size"):
            continue
        out.append({"name": m["name"], "disk_bytes": m["size"]})
    return sorted(out, key=lambda m: m["disk_bytes"])


_BLK = re.compile(r"^blk\.(\d+)\.attn_(k|v)\.weight$")
_BLK_NORM = re.compile(r"^blk\.(\d+)\.attn_k_norm\.weight$")


def kv_layout(tensors, key_length_swa=None):
    """Bucket the text layers by KV cache shape.

    Returns a list of buckets, each {n_layer, bytes_per_token, swa}. bytes_per_token is
    K plus V at two bytes per element, which is Ollama's default f16 cache.
    """
    k_dim, v_dim, head_dim = {}, {}, {}
    for t in tensors:
        m = _BLK.match(t["name"])
        if m:
            layer, which = int(m.group(1)), m.group(2)
            (k_dim if which == "k" else v_dim)[layer] = t["shape"][-1]
            continue
        m = _BLK_NORM.match(t["name"])
        if m:
            head_dim[int(m.group(1))] = t["shape"][-1]

    buckets = collections.Counter()
    for layer, kd in k_dim.items():
        vd = v_dim.get(layer, kd)
        hd = head_dim.get(layer)
        swa = key_length_swa is not None and hd == key_length_swa
        buckets[((kd + vd) * 2, swa)] += 1
    return [
        {"n_layer": n, "bytes_per_token": bpt, "swa": swa}
        for (bpt, swa), n in sorted(buckets.items(), reverse=True)
    ]


def describe(name):
    d = _post("/api/show", {"model": name})
    mi = d.get("model_info", {})
    det = d.get("details", {})
    arch = mi.get("general.architecture", "unknown")

    def a(key, default=None):
        v = mi.get(f"{arch}.{key}", default)
        return default if v is None else v

    layout = kv_layout(d.get("tensors", []), a("attention.key_length_swa"))
    window = a("attention.sliding_window", 0)

    return {
        "name": name,
        "arch": arch,
        "params": mi.get("general.parameter_count", 0),
        "quant": det.get("quantization_level", "unknown"),
        "n_layer": a("block_count", 0),
        "n_embd": a("embedding_length", 0),
        "n_head": a("attention.head_count", 0),
        "n_expert": a("expert_count", 0),
        "is_moe": 1 if a("expert_count", 0) else 0,
        "is_hybrid": 1 if a("ssm.state_size", 0) else 0,
        "vocab": a("vocab_size", 0) or len(d.get("tokenizer", {}) or {}) or 0,
        "train_ctx": a("context_length", 0),
        "sliding_window": window,
        "kv_layout": layout,
        "n_attn_layer": sum(b["n_layer"] for b in layout),
        "kv_bytes_per_token_global": sum(
            b["bytes_per_token"] * b["n_layer"] for b in layout if not b["swa"]
        ),
        "kv_bytes_per_token_swa": sum(
            b["bytes_per_token"] * b["n_layer"] for b in layout if b["swa"]
        ),
    }


def kv_mib(info, ctx):
    """Predicted KV cache size in MiB at this context, honouring sliding windows."""
    swa_tokens = min(ctx, info["sliding_window"]) if info["sliding_window"] else 0
    return (
        info["kv_bytes_per_token_global"] * ctx
        + info["kv_bytes_per_token_swa"] * swa_tokens
    ) / MIB


def catalog():
    """Every local model, with metadata and disk size, keyed by name."""
    out = {}
    for m in list_models():
        info = describe(m["name"])
        info["disk_bytes"] = m["disk_bytes"]
        info["weights_mib"] = m["disk_bytes"] / MIB
        out[m["name"]] = info
    return out


if __name__ == "__main__":
    print(json.dumps(catalog(), indent=2))
