import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class RepositoryContractTests(unittest.TestCase):
    def test_dataset_counts(self):
        d = json.loads((ROOT / "metrics/metric_registry.json").read_text())
        self.assertEqual((d["dataset"]["images"], d["dataset"]["classes"]), (109107, 120))
        self.assertEqual(d["dataset_audit"]["final_crossing_leakage_groups"], 0)

    def test_model_states(self):
        rows = list(csv.DictReader((ROOT / "metrics/main_results.csv").open()))
        self.assertEqual({r["state"] for r in rows}, {"reference", "mobile_float", "converted_int8", "pte_runtime"})

    def test_pte_identity(self):
        d = json.loads((ROOT / "metrics/metric_registry.json").read_text())
        self.assertEqual(d["pte"]["bytes"], 23696352)
        self.assertEqual(d["pte"]["sha256"], "7c70d0f307f0d9578310600913cb8ff171b294ae4de0813f7cd1d6a53628bdf1")

if __name__ == "__main__":
    unittest.main()
