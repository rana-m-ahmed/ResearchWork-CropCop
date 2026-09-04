import csv
import json
import unittest
from collections import defaultdict
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

    def test_historical_complement_is_plantcity_plus_bangladesh(self):
        rows = {r["ingestion_registry_key"]: r for r in csv.DictReader((ROOT / "data_card/provenance/historical_test_complement.csv").open())}
        deltas = {k:int(v["manifest_to_v4_delta"]) for k,v in rows.items() if int(v["manifest_to_v4_delta"]) != 0}
        self.assertEqual(deltas, {"plantcity_pk": 26017, "bangladesh": 1})
        self.assertEqual(sum(deltas.values()), 26018)
        self.assertEqual(int(rows["rice"]["manifest_to_v4_delta"]), 0)

    def test_registry_to_v4_attrition_is_16_rice_rows(self):
        rows = {r["registry_key"]: r for r in csv.DictReader((ROOT / "data_card/lineage/source_registry.csv").open())}
        deltas = {k:int(v["rows_in_recovered_registry"])-int(v["rows_in_v4_audited_universe"]) for k,v in rows.items()}
        self.assertEqual({k:v for k,v in deltas.items() if v}, {"rice": 16})

    def test_final_accounting_and_certificate(self):
        cert = json.loads((ROOT / "evidence/public/dataset_lineage/final_dataset_certificate.json").read_text())
        self.assertEqual(cert["source_images"], 109150)
        self.assertEqual(cert["manual_review_deleted"], 43)
        self.assertEqual(cert["final_images"], 109107)
        self.assertEqual(cert["split_counts"], {"test": 16363, "train": 76376, "val": 16368})
        rows = list(csv.DictReader((ROOT / "data_card/audit/manual_review_summary.csv").open()))
        m = {(r["dimension"], r["value"]): int(r["deleted_rows"]) for r in rows}
        self.assertEqual((m[("split", "train")], m[("split", "val")], m[("split", "test")]), (29, 8, 6))

    def test_provenance_confidence_is_bounded(self):
        rows = list(csv.DictReader((ROOT / "data_card/provenance/provenance_coverage.csv").open()))
        self.assertEqual(sum(int(r["final_rows"]) for r in rows), 109107)
        self.assertEqual(sum(int(r["final_rows"]) for r in rows if r["original_source_path_status"] == "available"), 84146)
        unknown = [r for r in rows if r["original_source_path_status"] == "unknown"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["ingestion_registry_key"], "plantcity_pk")
        self.assertEqual(int(unknown[0]["final_rows"]), 24961)
        self.assertEqual(unknown[0]["provenance_method"], "registry_complement_reconstruction")
        self.assertEqual(unknown[0]["provenance_confidence"], "reconstructed_unique_complement")
        defs = json.loads((ROOT / "data_card/provenance/provenance_confidence_definitions.json").read_text())
        self.assertEqual(defs["supersession"]["prior_public_label"], "verified_exact_source_family")

    def test_source_counts_reconcile_per_family(self):
        reg = {r["registry_key"]: r for r in csv.DictReader((ROOT / "data_card/lineage/source_registry.csv").open())}
        counts = {r["ingestion_registry_key"]: r for r in csv.DictReader((ROOT / "data_card/provenance/final_source_counts.csv").open())}
        splits = {r["ingestion_registry_key"]: r for r in csv.DictReader((ROOT / "data_card/provenance/final_source_split_counts.csv").open())}
        prov = defaultdict(int)
        for r in csv.DictReader((ROOT / "data_card/provenance/provenance_coverage.csv").open()):
            prov[r["ingestion_registry_key"]] += int(r["final_rows"])
        final_keys = {k for k,r in reg.items() if int(r["rows_in_final_frozen_dataset"]) > 0}
        self.assertEqual(final_keys, set(counts))
        self.assertEqual(final_keys, set(splits))
        self.assertEqual(final_keys, set(prov))
        for k in final_keys:
            vals = {
                int(reg[k]["rows_in_final_frozen_dataset"]),
                int(counts[k]["final_rows"]),
                sum(int(splits[k][s]) for s in ("train","val","test")),
                prov[k],
            }
            self.assertEqual(len(vals), 1, k)

    def test_external_exclusion_registry_covers_all_historical_sources(self):
        reg = {r["registry_key"] for r in csv.DictReader((ROOT / "data_card/lineage/source_registry.csv").open())}
        rows = list(csv.DictReader((ROOT / "data_card/provenance/external_evaluation_exclusion_registry.csv").open()))
        self.assertEqual(reg, {r["registry_key"] for r in rows})
        self.assertTrue(all(r["external_evaluation_status"] == "exclude_pending_independence_proof" for r in rows))

    def test_ontology_transition_registry(self):
        rows = {r["transition_id"]: r for r in csv.DictReader((ROOT / "data_card/lineage/ontology_transition_registry.csv").open())}
        self.assertEqual(set(rows), {"OT01","OT02","OT03","OT04","OT05","OT06","OT07"})
        self.assertIn("Smut", rows["OT02"]["details"])
        self.assertIn("Healthy", rows["OT02"]["details"])
        self.assertIn("rice_neck_blast", rows["OT07"]["details"])

    def test_fingerprint_scope_is_not_overclaimed(self):
        status = json.loads((ROOT / "data_card/audit/fingerprint_verification_status.json").read_text())
        rows = {r["id"]: r for r in status["fingerprints"]}
        self.assertFalse(rows["final_manifest_fingerprint"]["independent_public_recomputation"])
        self.assertFalse(rows["final_build_fingerprint"]["independent_public_recomputation"])
        self.assertEqual(rows["final_manifest_fingerprint"]["public_validator_status"], "cross_file_consistency_only")

    def test_manual_review_reconstruction(self):
        r = json.loads((ROOT / "data_card/audit/model_readiness_manual_review_reconstruction.json").read_text())
        self.assertEqual((r["residual_cross_label_similarity_pairs"], r["unique_pair_endpoints"], r["manual_review_queue_rows"]), (122, 180, 180))
        self.assertTrue(r["pair_endpoint_set_equals_manual_review_queue"])
        self.assertTrue(r["priority_score_formula_matches_all_180_rows"])
        self.assertEqual(r["selector_source_code_status"], "not supplied in the recovered evidence bundle")

if __name__ == "__main__":
    unittest.main()
