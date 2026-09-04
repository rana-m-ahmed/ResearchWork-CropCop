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

    def test_duplicate_edge_semantics(self):
        rows = {r["edge_category"]: r for r in csv.DictReader((ROOT / "data_card/audit/duplicate_edge_history.csv").open())}
        self.assertEqual(int(rows["historical_confirmed_total"]["historical_v4_count"]), 8672)
        self.assertEqual(int(rows["historical_confirmed_total"]["corrected_v5_count"]), 8573)
        self.assertEqual(int(rows["confirmed_cross_label_subset"]["corrected_v5_count"]), 17)
        self.assertEqual(6196 + 445 + 1932, 8573)

    def test_final_accounting_and_certificate(self):
        cert = json.loads((ROOT / "evidence/public/dataset_lineage/final_dataset_certificate.json").read_text())
        self.assertEqual(cert["source_images"], 109150)
        self.assertEqual(cert["manual_review_deleted"], 43)
        self.assertEqual(cert["final_images"], 109107)
        self.assertEqual(cert["split_counts"], {"test": 16363, "train": 76376, "val": 16368})
        rows = list(csv.DictReader((ROOT / "data_card/audit/manual_review_summary.csv").open()))
        m = {(r["dimension"], r["value"]): int(r["deleted_rows"]) for r in rows}
        self.assertEqual((m[("split", "train")], m[("split", "val")], m[("split", "test")]), (29, 8, 6))

    def test_provenance_coverage(self):
        rows = list(csv.DictReader((ROOT / "data_card/provenance/provenance_coverage.csv").open()))
        self.assertEqual(sum(int(r["final_rows"]) for r in rows), 109107)
        self.assertEqual(sum(int(r["final_rows"]) for r in rows if r["original_source_path_status"] == "available"), 84146)
        self.assertEqual(sum(int(r["final_rows"]) for r in rows if r["original_source_path_status"] == "unknown"), 24961)
        self.assertEqual({r["ingestion_registry_key"] for r in rows if r["original_source_path_status"] == "unknown"}, {"plantcity_pk"})

    def test_manual_review_reconstruction(self):
        r = json.loads((ROOT / "data_card/audit/model_readiness_manual_review_reconstruction.json").read_text())
        self.assertEqual((r["residual_cross_label_similarity_pairs"], r["unique_pair_endpoints"], r["manual_review_queue_rows"]), (122, 180, 180))
        self.assertTrue(r["pair_endpoint_set_equals_manual_review_queue"])
        self.assertTrue(r["priority_score_formula_matches_all_180_rows"])

if __name__ == "__main__":
    unittest.main()
