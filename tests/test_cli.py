import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "quantweavelm/data/demo_config.json"
DATA = ROOT / "quantweavelm/data/demo.jsonl"


class CliTests(unittest.TestCase):
    def invoke(self, *arguments, check=True):
        return subprocess.run([sys.executable, "-m", "quantweavelm", *map(str, arguments)],
                              cwd=ROOT, text=True, capture_output=True, check=check)

    def test_demo_is_plain_deterministic_output(self):
        first = self.invoke("demo").stdout
        second = self.invoke("demo").stdout
        self.assertEqual(first, second)
        self.assertIn("mode=offline_research_only no_profit_claim", first)
        self.assertNotIn("\x1b", first)

    def test_run_verify_summary_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            prompt = Path(directory) / "prompt.json"
            self.invoke("run", CONFIG, DATA, report)
            self.assertEqual(self.invoke("verify", CONFIG, DATA, report).stdout, "report verified\n")
            self.assertIn("test calibrated log_loss", self.invoke("summary", report).stdout)
            self.invoke("prompt", report, prompt)
            self.assertEqual(json.loads(prompt.read_text())["messages"][0]["role"], "system")
            self.assertTrue(prompt.read_bytes().endswith(b"\n"))

    def test_failure_has_nonzero_exit_and_no_traceback(self):
        result = self.invoke("run", CONFIG, DATA.with_name("missing.jsonl"), "out.json", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("quantweavelm:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
