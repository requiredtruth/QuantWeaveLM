import math
import unittest

from quantweavelm.model import (aggregate, apply_temperature, mixture,
                                multiclass_brier, ranked_probability_score)


class ModelTests(unittest.TestCase):
    def test_linear_pool_and_temperature_are_normalized(self):
        forecasts = {"a": [0.8, 0.1, 0.1], "b": [0.2, 0.3, 0.5]}
        pooled = mixture(forecasts, ["a", "b"], [0.25, 0.75])
        self.assertAlmostEqual(sum(pooled), 1.0)
        self.assertEqual([round(value, 3) for value in pooled], [0.35, 0.25, 0.4])
        softened = apply_temperature(pooled, 2.0)
        self.assertAlmostEqual(sum(softened), 1.0)
        self.assertLess(max(softened) - min(softened), max(pooled) - min(pooled))

    def test_ordered_score_penalizes_farther_miss(self):
        near = ranked_probability_score([0.0, 1.0, 0.0], 0)
        far = ranked_probability_score([0.0, 0.0, 1.0], 0)
        self.assertLess(near, far)
        self.assertEqual(multiclass_brier([1.0, 0.0, 0.0], 0), 0.0)

    def test_aggregate_metrics_are_finite_and_explicit(self):
        result = aggregate([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]], [0, 2])
        for key in ("log_loss", "multiclass_brier", "ranked_probability_score",
                    "top_label_accuracy", "top_label_ece"):
            self.assertTrue(math.isfinite(result[key]))
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["classwise_reliability"]), 3)


if __name__ == "__main__":
    unittest.main()
