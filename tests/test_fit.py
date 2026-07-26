"""The solver, and the quality of the fit that ships in model.json."""

import json
import pathlib
import unittest

from vram_oracle import fit as fitmod

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_JSON = ROOT / "model.json"
DATA = ROOT / "data" / "measurements.jsonl"


class TestSolver(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        import random

        rng = random.Random(7)
        truth = [500.0, 1.02, 0.98, 3.5, 2.0, -120.0]
        X, y = [], []
        for _ in range(40):
            r = [
                1.0,
                rng.uniform(4000, 24000),
                rng.uniform(0, 9000),
                float(rng.randint(24, 64)),
                rng.uniform(2, 6),
                float(rng.randint(0, 1)),
            ]
            X.append(r)
            y.append(sum(c * v for c, v in zip(truth, r)))
        got = fitmod.solve(X, y, ridge=0.0)
        for a, b in zip(got, truth):
            self.assertAlmostEqual(a, b, places=4)

    def test_singular_design_is_an_error_not_a_silent_zero(self):
        X = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
        with self.assertRaises(ValueError):
            fitmod.solve(X, [1.0, 2.0, 3.0], ridge=0.0)

    def test_r_squared_is_one_for_an_exact_fit(self):
        y = [1.0, 5.0, 9.0]
        self.assertAlmostEqual(fitmod.r_squared(y, list(y)), 1.0)

    def test_error_summary(self):
        e = fitmod.errors([100.0, 200.0], [110.0, 180.0])
        self.assertEqual(e["n"], 2)
        self.assertAlmostEqual(e["mae_mib"], 15.0)
        self.assertAlmostEqual(e["max_abs_mib"], 20.0)
        self.assertAlmostEqual(e["bias_mib"], -5.0)


class TestFittedModel(unittest.TestCase):
    """Guards the shipped fit against silent regression when data or features change."""

    @classmethod
    def setUpClass(cls):
        if not MODEL_JSON.exists():
            raise unittest.SkipTest("no model.json yet; run vram-oracle fit")
        cls.m = json.loads(MODEL_JSON.read_text())

    def test_holdout_is_large_enough_to_mean_something(self):
        self.assertGreaterEqual(self.m["holdout"]["n"], 8)

    def test_holdout_error_matches_the_advertised_error_bar(self):
        self.assertLessEqual(self.m["holdout"]["max_abs_mib"], self.m["error_bar_mib"] + 1)

    def test_holdout_accuracy_target(self):
        """The catalog task asks for predictions within 1 GiB on held-out configs."""
        self.assertLessEqual(self.m["holdout"]["max_abs_mib"], 1024)

    def test_fit_is_physically_sensible(self):
        c = self.m["coefficients"]
        self.assertAlmostEqual(c["weights_mib"], 1.0, delta=0.1)
        self.assertAlmostEqual(c["kv_mib"], 1.0, delta=0.25)
        self.assertGreater(c["intercept"], 0)

    def test_refit_reproduces_the_shipped_coefficients(self):
        if not DATA.exists():
            self.skipTest("no measurements checked in")
        samples, _ = fitmod.load_samples(DATA, self.m["catalog"])
        self.assertEqual(len(samples), self.m["n_measurements"])
        # Refit with the features the shipped model actually selected. Calling fit()
        # without them silently falls back to the full default feature list, which
        # produces different coefficients for a legitimate reason and made this test
        # look like a reproducibility failure when nothing was wrong.
        features = self.m["features"]
        coef = fitmod.fit(samples, self.m["catalog"], None, features)
        for k, v in zip(features, coef):
            self.assertAlmostEqual(v, self.m["coefficients"][k], places=6)

    def test_holdout_split_is_deterministic(self):
        samples, _ = fitmod.load_samples(DATA, self.m["catalog"])
        a = fitmod.split(samples)
        b = fitmod.split(samples)
        self.assertEqual(a, b)

    def test_predictions_increase_with_context(self):
        info = next(iter(self.m["catalog"].values()))
        coef = [self.m["coefficients"][f] for f in self.m["features"]]
        vals = [fitmod.predict(coef, info, c) for c in (2048, 8192, 32768, 131072)]
        self.assertEqual(vals, sorted(vals))


if __name__ == "__main__":
    unittest.main()
