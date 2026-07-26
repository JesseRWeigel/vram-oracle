"""KV layout parsing. These run without Ollama; the tensor tables are synthetic."""

import unittest

from vram_oracle.model_info import kv_layout, kv_mib

MIB = 1024 * 1024


def tensors(spec):
    """spec: list of (n_layers, k_out_dim, head_dim). Returns a fake tensor table."""
    out, layer = [], 0
    for n, kd, hd in spec:
        for _ in range(n):
            out.append({"name": f"blk.{layer}.attn_k.weight", "shape": [4096, kd]})
            out.append({"name": f"blk.{layer}.attn_v.weight", "shape": [4096, kd]})
            out.append({"name": f"blk.{layer}.attn_k_norm.weight", "shape": [hd]})
            out.append({"name": f"blk.{layer}.ffn_up.weight", "shape": [4096, 11008]})
            layer += 1
    return out


class TestKVLayout(unittest.TestCase):
    def test_dense_model_single_bucket(self):
        layout = kv_layout(tensors([(36, 1024, 128)]))
        self.assertEqual(len(layout), 1)
        self.assertEqual(layout[0]["n_layer"], 36)
        self.assertEqual(layout[0]["bytes_per_token"], (1024 + 1024) * 2)
        self.assertFalse(layout[0]["swa"])

    def test_hybrid_stack_counts_only_attention_layers(self):
        # 8 attention layers interleaved into a 32 layer stack; the SSM layers have no
        # attn_k tensor at all, so they must not contribute to the KV cache.
        t = tensors([(8, 1024, 128)])
        t += [{"name": f"blk.{i}.ssm_in.weight", "shape": [4096, 4096]} for i in range(8, 32)]
        layout = kv_layout(t)
        self.assertEqual(sum(b["n_layer"] for b in layout), 8)

    def test_sliding_window_bucket_detected_by_head_dim(self):
        # gemma4 shape: 50 windowed layers at head_dim 256, 10 global layers at 512.
        layout = kv_layout(tensors([(50, 4096, 256), (10, 2048, 512)]), key_length_swa=256)
        swa = [b for b in layout if b["swa"]]
        glob = [b for b in layout if not b["swa"]]
        self.assertEqual(swa[0]["n_layer"], 50)
        self.assertEqual(glob[0]["n_layer"], 10)

    def test_vision_tower_tensors_ignored(self):
        t = tensors([(4, 1024, 128)])
        t += [{"name": f"v.blk.{i}.attn_k.weight", "shape": [1152, 1152]} for i in range(27)]
        layout = kv_layout(t)
        self.assertEqual(sum(b["n_layer"] for b in layout), 4)


class TestKVSize(unittest.TestCase):
    def info(self, glob, swa, window):
        return {
            "kv_bytes_per_token_global": glob,
            "kv_bytes_per_token_swa": swa,
            "sliding_window": window,
        }

    def test_global_cache_is_linear_in_context(self):
        i = self.info(4096, 0, 0)
        self.assertAlmostEqual(kv_mib(i, 8192), 4096 * 8192 / MIB)
        self.assertAlmostEqual(kv_mib(i, 16384), 2 * kv_mib(i, 8192))

    def test_windowed_cache_saturates_at_the_window(self):
        i = self.info(0, 8192, 1024)
        at_1k = kv_mib(i, 1024)
        self.assertAlmostEqual(kv_mib(i, 131072), at_1k)
        self.assertAlmostEqual(kv_mib(i, 512), at_1k / 2)


if __name__ == "__main__":
    unittest.main()
