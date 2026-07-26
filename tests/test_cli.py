"""CLI behaviour, driven by a synthetic fitted model so the numbers are controlled.

Ollama is not required: passing a model name that is in the fixture catalog but not on
the daemon exercises the cached-catalog path, and --free-mib replaces the GPU query.
"""

import io
import json
import contextlib
import pathlib
import tempfile
import unittest

from vram_oracle import cli

FIXTURE = {
    "features": ["intercept", "weights_mib", "kv_mib", "n_layer", "activation_mib", "is_moe"],
    "coefficients": {
        "intercept": 400.0, "weights_mib": 1.0, "kv_mib": 1.0,
        "n_layer": 0.0, "activation_mib": 0.0, "is_moe": 0.0,
    },
    "error_bar_mib": 500,
    "n_measurements": 40,
    "holdout": {"n": 8, "max_abs_mib": 500.0},
    "catalog": {
        "fixture:10b": {
            "name": "fixture:10b", "arch": "test", "weights_mib": 10000.0,
            "n_layer": 32, "n_embd": 4096, "is_moe": 0, "n_attn_layer": 32,
            "kv_bytes_per_token_global": 128 * 1024, "kv_bytes_per_token_swa": 0,
            "sliding_window": 0, "train_ctx": 131072,
        },
    },
}


def run(args, model_json):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(["--model-json", str(model_json)] + args)
    return code, buf.getvalue()


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mj = pathlib.Path(cls.tmp.name) / "model.json"
        cls.mj.write_text(json.dumps(FIXTURE))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prediction_matches_the_arithmetic(self):
        # 10000 weights + 400 intercept + 8192 tokens * 128 KiB = 1024 MiB of cache
        code, out = run(["predict", "fixture:10b", "--ctx", "8192", "--json"], self.mj)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["predicted_mib"], 11424)

    def test_verdict_fits_when_worst_case_clears_free_vram(self):
        code, out = run(
            ["will-it-fit", "fixture:10b", "--ctx", "8192", "--free-mib", "12000", "--json"],
            self.mj)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["verdict"], "FITS")

    def test_verdict_tight_inside_the_error_bar(self):
        # predicted 11424, worst case 11924: 11500 free is enough only if the fit is right
        code, out = run(
            ["will-it-fit", "fixture:10b", "--ctx", "8192", "--free-mib", "11500", "--json"],
            self.mj)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["verdict"], "TIGHT")

    def test_verdict_will_not_fit(self):
        code, out = run(
            ["will-it-fit", "fixture:10b", "--ctx", "8192", "--free-mib", "8000", "--json"],
            self.mj)
        self.assertEqual(code, 2)
        d = json.loads(out)
        self.assertEqual(d["verdict"], "WILL NOT FIT")
        self.assertEqual(d["headroom_mib"], 8000 - 11424)

    def test_context_growth_flips_the_verdict(self):
        fits, _ = run(["will-it-fit", "fixture:10b", "--ctx", "2048", "--free-mib", "13000"], self.mj)
        nope, _ = run(["will-it-fit", "fixture:10b", "--ctx", "131072", "--free-mib", "13000"], self.mj)
        self.assertEqual(fits, 0)
        self.assertEqual(nope, 2)

    def test_max_ctx_lands_on_the_boundary(self):
        # 20000 free, 500 error bar: cache budget is 20000-500-10400 = 9100 MiB at
        # 0.125 MiB per token, so about 72800 tokens.
        code, out = run(["max-ctx", "fixture:10b", "--free-mib", "20000"], self.mj)
        self.assertEqual(code, 0)
        ctx = int(out.split("up to ctx ")[1].split(" ")[0].replace(",", ""))
        self.assertGreater(ctx, 72000)
        self.assertLess(ctx, 73000)

    def test_max_ctx_reports_no_fit_at_all(self):
        code, out = run(["max-ctx", "fixture:10b", "--free-mib", "4000"], self.mj)
        self.assertEqual(code, 2)
        self.assertIn("does not fit", out)

    def test_explain_shows_every_term(self):
        _, out = run(["predict", "fixture:10b", "--ctx", "8192", "--explain"], self.mj)
        for term in FIXTURE["features"]:
            self.assertIn(term, out)

    def test_table_marks_configs_that_do_not_fit(self):
        _, out = run(["table", "--contexts", "2048,131072", "--free-mib", "12000"], self.mj)
        self.assertIn("fixture:10b", out)
        self.assertIn("!", out)


if __name__ == "__main__":
    unittest.main()
