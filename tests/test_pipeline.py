import copy
import json
import unittest
from pathlib import Path

from quantweavelm.core import QuantWeaveError
from quantweavelm.pipeline import run_pipeline, validate_config, validate_rows
from quantweavelm.report import prompt, verify

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "quantweavelm/data/demo_config.json").read_text())
ROWS = [json.loads(line) for line in (ROOT / "quantweavelm/data/demo.jsonl").read_text().splitlines()]


class PipelineTests(unittest.TestCase):
    def test_demo_pipeline_has_declared_boundaries(self):
        report = run_pipeline(CONFIG, ROWS)
        self.assertFalse(report["claims"]["test_used_for_fitting"])
        self.assertFalse(report["claims"]["embargo_rows_used_for_fitting"])
        self.assertEqual(sum(report["learned"]["weights"].values()), 1.0)
        self.assertEqual(report["metrics"]["test"]["calibrated_pool"]["count"], 10)
        self.assertIn("calibration_climatology", report["metrics"]["test"])

    def test_test_targets_cannot_change_learned_parameters(self):
        baseline = run_pipeline(CONFIG, ROWS)
        changed = copy.deepcopy(ROWS)
        changed[-1]["target_return"] = -0.2
        rerun = run_pipeline(CONFIG, changed)
        self.assertEqual(baseline["learned"], rerun["learned"])
        self.assertNotEqual(baseline["evidence"]["dataset_sha256"], rerun["evidence"]["dataset_sha256"])

    def test_embargo_forecasts_cannot_change_learning_or_metrics(self):
        baseline = run_pipeline(CONFIG, ROWS)
        changed = copy.deepcopy(ROWS)
        changed[20]["forecasts"]["momentum"] = [1.0, 0.0, 0.0]
        rerun = run_pipeline(CONFIG, changed)
        self.assertEqual(baseline["learned"], rerun["learned"])
        self.assertEqual(baseline["metrics"], rerun["metrics"])

    def test_malformed_probability_and_timestamp_fail_closed(self):
        invalid = copy.deepcopy(ROWS)
        invalid[0]["forecasts"]["momentum"] = [0.2, 0.2, 0.2]
        with self.assertRaisesRegex(QuantWeaveError, "sum to 1"):
            run_pipeline(CONFIG, invalid)
        invalid = copy.deepcopy(ROWS)
        invalid[1]["timestamp"] = invalid[0]["timestamp"]
        with self.assertRaisesRegex(QuantWeaveError, "strictly increasing"):
            run_pipeline(CONFIG, invalid)

    def test_split_counts_and_embargo_time_are_enforced(self):
        with self.assertRaisesRegex(QuantWeaveError, "exactly 32"):
            run_pipeline(CONFIG, ROWS[:-1])
        config = copy.deepcopy(CONFIG)
        config["split"]["minimum_embargo_seconds"] = 1_000_000
        with self.assertRaisesRegex(QuantWeaveError, "timestamp embargo"):
            run_pipeline(config, ROWS)

    def test_verify_detects_report_tampering(self):
        report = run_pipeline(CONFIG, ROWS)
        self.assertIs(verify(CONFIG, ROWS, report), report)
        changed = copy.deepcopy(report)
        changed["learned"]["temperature"] = 1.0
        with self.assertRaisesRegex(QuantWeaveError, "recomputation"):
            verify(CONFIG, ROWS, changed)

    def test_prompt_is_bounded_and_contains_no_rows_or_timestamps(self):
        material = json.dumps(prompt(run_pipeline(CONFIG, ROWS)))
        self.assertNotIn(ROWS[0]["timestamp"], material)
        self.assertNotIn("target_return", material)
        self.assertIn("Do not predict prices", material)

    def test_config_rejects_unknown_fields(self):
        invalid = copy.deepcopy(CONFIG)
        invalid["symbol"] = "EXAMPLE"
        with self.assertRaises(QuantWeaveError):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
