import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class ReleaseTests(unittest.TestCase):
    def test_support_addresses_and_terms_are_exact(self):
        support = (ROOT / "SUPPORT.md").read_text()
        addresses = ["bc1qh474jpyw4malh0fmg2uy7n05ggtjvnjtcwhdne",
                     "0x8fcC9C0d1FFCE17b1dEC91B299E56d66BC126Ba8",
                     "D6qp2awRAHVo2VgincTAW5frhnJ9MBZcz4"]
        for value in addresses:
            self.assertEqual(support.count(value), 1)
        self.assertIn("does not purchase support, ownership, investment returns", support)

    def test_public_tree_contains_no_private_project_marker(self):
        marker = "World" + "Forge"
        for path in ROOT.rglob("*"):
            if (path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
                    and "build" not in path.parts and not any(part.endswith(".egg-info") for part in path.parts)
                    and path.name != "LICENSE"):
                self.assertNotIn(marker, path.read_text(errors="ignore"), str(path))

    def test_package_has_no_runtime_dependency_and_demo_works(self):
        project = (ROOT / "pyproject.toml").read_text()
        self.assertIn("dependencies = []", project)
        result = subprocess.run([sys.executable, "-m", "quantweavelm", "demo"], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        self.assertIn("offline_research_only", result.stdout)

    def test_readme_links_and_disclosures_exist(self):
        readme = (ROOT / "README.md").read_text()
        for value in ("scikit-learn.org/stable/modules/calibration.html",
                      "doi.org/10.1198/016214506000001437", "ranked probability score",
                      "does not imply a profitable", "no API credentials"):
            self.assertIn(value, readme)


if __name__ == "__main__":
    unittest.main()
