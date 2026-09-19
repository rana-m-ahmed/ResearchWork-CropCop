from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.trackb_r07 import (
    CANDIDATE_CONTRACTS,
    REQUIRED_INPUT_ROLES,
    DINO_FACTORY_MANIFEST_SHA256,
    R07_CHECKPOINTS,
    R07_RUN_RECORDS,
    TrackBError,
    assign_candidate_grade,
    build_candidate_seal,
    mapped_scope_metrics,
    validate_same_prediction_surface,
    verify_candidate_seal,
    git_blob_sha1,
    verify_code_attestation,
    load_json,
    validate_execution_lock,
)
from cropcop_je.hashing import sha256_json
from cropcop_je.trackb_r07_analysis import bootstrap_three_seed_macro_f1
from cropcop_je.trackb_r07_audit import ImageAuditRecord, representative_manifest
from cropcop_je.trackb_r07_ops import load_kaggle_secret


class TrackBR07Tests(unittest.TestCase):
    def test_ext_i_requires_full_independence(self):
        grade = assign_candidate_grade(
            source_identity_ok=True,
            mapping_ok=True,
            family_support={"a": 50, "b": 80, "c": 60},
            required_labels=["a", "b", "c"],
            historical_surface_complete=True,
            accepted_historical_link_count=0,
            unresolved_lineage=False,
        )
        self.assertEqual(grade.grade, "EXT-I")

    def test_any_historical_link_permanently_blocks_ext_i(self):
        grade = assign_candidate_grade(
            source_identity_ok=True,
            mapping_ok=True,
            family_support={"a": 100, "b": 100, "c": 100},
            required_labels=["a", "b", "c"],
            historical_surface_complete=True,
            accepted_historical_link_count=1,
            unresolved_lineage=False,
        )
        self.assertEqual(grade.grade, "EXT-S")
        self.assertIn("ACCEPTED_HISTORICAL_LINKS", grade.reasons)

    def test_partial_historical_surface_caps_at_ext_s(self):
        grade = assign_candidate_grade(
            source_identity_ok=True,
            mapping_ok=True,
            family_support={"a": 100, "b": 100, "c": 100},
            required_labels=["a", "b", "c"],
            historical_surface_complete=False,
            accepted_historical_link_count=0,
            unresolved_lineage=False,
        )
        self.assertEqual(grade.grade, "EXT-S")

    def test_support_floor_blocks_claim_producing_inference(self):
        grade = assign_candidate_grade(
            source_identity_ok=True,
            mapping_ok=True,
            family_support={"a": 49, "b": 100, "c": 100},
            required_labels=["a", "b", "c"],
            historical_surface_complete=True,
            accepted_historical_link_count=0,
            unresolved_lineage=False,
        )
        self.assertEqual(grade.grade, "EXT-X")
        self.assertEqual(grade.claim_mode, "NO_PERFORMANCE")

    def test_mapping_failure_is_ext_x(self):
        grade = assign_candidate_grade(
            source_identity_ok=True,
            mapping_ok=False,
            family_support={"a": 100},
            required_labels=["a"],
            historical_surface_complete=True,
            accepted_historical_link_count=0,
            unresolved_lineage=False,
        )
        self.assertEqual(grade.grade, "EXT-X")


    def test_known_historical_contributor_is_ext_x(self):
        grade = assign_candidate_grade(
            source_identity_ok=True, mapping_ok=True, family_support={"a": 100, "b": 100, "c": 100},
            required_labels=["a", "b", "c"], historical_surface_complete=True,
            accepted_historical_link_count=0, unresolved_lineage=False,
            known_historical_contributor_relationship=True,
        )
        self.assertEqual(grade.grade, "EXT-X")

    def test_native_120_way_outside_mapped_prediction_is_error(self):
        rows = [
            {"stable_row_id": "1", "target_class_index": 1, "predicted_class_index": 1},
            {"stable_row_id": "2", "target_class_index": 1, "predicted_class_index": 119},
            {"stable_row_id": "3", "target_class_index": 2, "predicted_class_index": 2},
            {"stable_row_id": "4", "target_class_index": 2, "predicted_class_index": 1},
        ]
        metrics = mapped_scope_metrics(rows, [1, 2])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["out_of_mapped_subset_prediction_rate"], 0.25)
        self.assertLess(metrics["macro_f1"], 1.0)

    def _valid_claim_seal_payload(self):
        h = "a" * 64
        payload = {
            "candidate_id": "irish_potato",
            "scope": "SCOPE-POTATO-3",
            "source_doi": "10.5281/zenodo.8286529",
            "source_version": "01",
            "source_identity_ok": True,
            "mapping_ok": True,
            "unresolved_lineage": False,
            "known_historical_contributor_relationship": False,
            "mapping": {
                "earlyblt": "potato_early_blight",
                "healthy": "potato_healthy",
                "lateblt": "potato_late_blight",
            },
            "grade": "EXT-I",
            "claim_mode": "AUDIT_BOUNDED_SOURCE_INDEPENDENT",
            "grade_reasons": [],
            "sealed_at_utc": "2026-09-19T00:00:00+00:00",
            "prediction_count_at_seal": 0,
            "preprocessing": {
                "entrypoint": "cropcop_je.data.ctc_v2_eval_transform",
                "config_sha256": "53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4",
            },
            "bootstrap_seed": 409883112,
            "bootstrap_replicates": 5000,
            "external_family_order_seed": 1936263114,
            "downstream_authority_sha256": h,
            "execution_lock_sha256": h,
            "source_manifest_sha256": h,
            "source_metadata_record_sha256": h,
            "mapping_sha256": "",
            "family_graph_sha256": h,
            "representative_manifest_sha256": h,
            "exclusion_ledger_sha256": h,
            "decode_failure_ledger_sha256": h,
            "historical_compare_input_manifest_sha256": h,
            "accepted_within_edges_sha256": h,
            "accepted_historical_edges_sha256": h,
            "historical_comparison_summary_sha256": h,
            "within_comparison_summary_sha256": h,
            "family_support": {"earlyblt": 50, "healthy": 50, "lateblt": 50},
            "historical_surface_complete": True,
            "accepted_historical_link_count": 0,
            "authorized_r07_checkpoint_sha256": R07_CHECKPOINTS,
        }
        payload["mapping_sha256"] = sha256_json(payload["mapping"])
        return payload

    def test_seal_requires_all_three_r07_states_for_claim_candidate(self):
        seal = build_candidate_seal(self._valid_claim_seal_payload())
        verify_candidate_seal(seal)
        tampered = dict(seal)
        tampered["authorized_r07_checkpoint_sha256"] = {"S1": R07_CHECKPOINTS["S1"]}
        tampered = build_candidate_seal(tampered)
        with self.assertRaises(TrackBError):
            verify_candidate_seal(tampered)

    def test_ext_i_seal_cannot_hide_historical_links(self):
        payload = self._valid_claim_seal_payload()
        payload["accepted_historical_link_count"] = 1
        with self.assertRaises(TrackBError):
            verify_candidate_seal(build_candidate_seal(payload))

    def test_unresolved_lineage_caps_at_ext_s(self):
        grade = assign_candidate_grade(
            source_identity_ok=True, mapping_ok=True, family_support={"a": 100, "b": 100, "c": 100},
            required_labels=["a", "b", "c"], historical_surface_complete=True,
            accepted_historical_link_count=0, unresolved_lineage=True,
        )
        self.assertEqual(grade.grade, "EXT-S")

    def test_seal_rejects_missing_required_support_label(self):
        payload = self._valid_claim_seal_payload()
        payload["family_support"] = {"earlyblt": 50, "healthy": 50}
        with self.assertRaises(TrackBError):
            verify_candidate_seal(build_candidate_seal(payload))

    def test_ext_i_seal_rejects_unresolved_lineage(self):
        payload = self._valid_claim_seal_payload()
        payload["unresolved_lineage"] = True
        with self.assertRaises(TrackBError):
            verify_candidate_seal(build_candidate_seal(payload))

    def test_metrics_emit_native120_confusion_evidence(self):
        rows = [
            {"stable_row_id": "1", "target_class_index": 1, "predicted_class_index": 1},
            {"stable_row_id": "2", "target_class_index": 1, "predicted_class_index": 119},
            {"stable_row_id": "3", "target_class_index": 2, "predicted_class_index": 2},
        ]
        metrics = mapped_scope_metrics(rows, [1, 2])
        self.assertEqual(metrics["confusion_matrix_native120_nonzero"]["1"], {"1": 1, "119": 1})

    def test_code_attestation_detects_source_drift(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f = root / "x.py"
            f.write_text("print('ok')\n", encoding="utf-8")
            att = {
                "attestation_id": "TRACKB_CODE_ATTESTATION_v2",
                "parent_track_a_closure_commit": "604aafd51e20e70098ce4af647e90c8ff558a9e8",
                "files": [{"path": "x.py", "git_blob_sha1": git_blob_sha1(f)}],
            }
            ap = root / "att.json"
            ap.write_text(json.dumps(att), encoding="utf-8")
            verify_code_attestation(root, ap)
            f.write_text("print('drift')\n", encoding="utf-8")
            with self.assertRaises(TrackBError):
                verify_code_attestation(root, ap)

    def test_v2_candidate_roles_are_frozen_and_agrivision_is_retired(self):
        self.assertEqual(REQUIRED_INPUT_ROLES, {"core", "historical_compare", "gvlid_v5", "irish_potato"})
        self.assertNotIn("agrivision_bd", CANDIDATE_CONTRACTS)
        self.assertEqual(
            CANDIDATE_CONTRACTS["gvlid_grape"]["mapping"],
            {
                "Black Rot": "grape_black_rot",
                "Esca": "grape_esca",
                "Healthy": "grape_healthy",
                "Leaf Blight": "grape_leaf_blight",
            },
        )

    def test_v2_lock_requires_automated_publication_boundaries(self):
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v2.json")
        validate_execution_lock(lock)
        automation = lock["automation"]
        self.assertTrue(automation["single_master_notebook"])
        self.assertTrue(automation["auto_commit_public_safe_evidence_to_github"])
        self.assertTrue(automation["auto_archive_complete_restricted_evidence_to_private_kaggle"])
        self.assertFalse(automation["raw_images_to_github"])
        self.assertFalse(automation["checkpoints_to_github"])
        self.assertEqual(automation["kaggle_dataset_owner_default"], "AUTO")
        self.assertEqual(
            automation["kaggle_dataset_owner_mode"],
            "AUTHENTICATED_TOKEN_OWNER_AUTO_DETECT",
        )
        self.assertTrue(lock["kaggle"]["internet_required_during_claim_run"])
        self.assertEqual(
            lock["kaggle"]["internet_role"],
            "ORCHESTRATION_AND_EVIDENCE_PUBLICATION_ONLY",
        )
        self.assertFalse(lock["kaggle"]["protected_inference_network_dependency"])
        self.assertEqual(lock["software"]["timm"], "1.0.26")
        bootstrap = lock["runtime_bootstrap"]
        self.assertEqual(bootstrap["mode"], "ISOLATED_VIRTUAL_ENVIRONMENT")
        self.assertEqual(bootstrap["pytorch_wheel_index"], "https://download.pytorch.org/whl/cu126")
        self.assertTrue(bootstrap["live_kernel_torch_replacement_forbidden"])
        self.assertEqual(
            bootstrap["secrets_handoff"],
            "PARENT_KAGGLE_KERNEL_TO_CHILD_ENVIRONMENT_ONLY",
        )

        drifted = dict(lock)
        drifted["automation"] = dict(automation)
        drifted["automation"]["raw_images_to_github"] = True
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

        drifted_owner = dict(lock)
        drifted_owner["automation"] = dict(automation)
        drifted_owner["automation"]["kaggle_dataset_owner_default"] = "someone"
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted_owner)

        drifted_network = dict(lock)
        drifted_network["kaggle"] = dict(lock["kaggle"])
        drifted_network["kaggle"]["protected_inference_network_dependency"] = True
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted_network)

    def test_secret_loader_prefers_environment_for_isolated_runtime(self):
        key = "TRACKB_TEST_SECRET"
        previous = os.environ.get(key)
        try:
            os.environ[key] = "sentinel-token"
            self.assertEqual(load_kaggle_secret(key), "sentinel-token")
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def test_gvlid_seal_contract_accepts_exact_four_class_scope(self):
        h = "b" * 64
        mapping = CANDIDATE_CONTRACTS["gvlid_grape"]["mapping"]
        payload = {
            "candidate_id": "gvlid_grape",
            "scope": "SCOPE-GRAPE-4",
            "source_doi": "10.17632/wkymf8bhcg.5",
            "source_version": "5",
            "source_identity_ok": True,
            "mapping_ok": True,
            "unresolved_lineage": True,
            "known_historical_contributor_relationship": False,
            "mapping": mapping,
            "grade": "EXT-S",
            "claim_mode": "CROSS_DATASET_STRESS",
            "grade_reasons": ["HISTORICAL_COMPARISON_INCOMPLETE", "UNRESOLVED_LINEAGE"],
            "sealed_at_utc": "2026-09-19T00:00:00+00:00",
            "prediction_count_at_seal": 0,
            "preprocessing": {
                "entrypoint": "cropcop_je.data.ctc_v2_eval_transform",
                "config_sha256": "53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4",
            },
            "bootstrap_seed": 409883112,
            "bootstrap_replicates": 5000,
            "external_family_order_seed": 1936263114,
            "downstream_authority_sha256": h,
            "execution_lock_sha256": h,
            "source_manifest_sha256": h,
            "source_metadata_record_sha256": h,
            "mapping_sha256": sha256_json(mapping),
            "family_graph_sha256": h,
            "representative_manifest_sha256": h,
            "exclusion_ledger_sha256": h,
            "decode_failure_ledger_sha256": h,
            "historical_compare_input_manifest_sha256": h,
            "accepted_within_edges_sha256": h,
            "accepted_historical_edges_sha256": h,
            "historical_comparison_summary_sha256": h,
            "within_comparison_summary_sha256": h,
            "family_support": {label: 50 for label in mapping},
            "historical_surface_complete": False,
            "accepted_historical_link_count": 0,
            "authorized_r07_checkpoint_sha256": R07_CHECKPOINTS,
        }
        verify_candidate_seal(build_candidate_seal(payload))

    def test_execution_lock_binds_replay_records_and_dino_factory(self):
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v2.json")
        validate_execution_lock(lock)
        ids = lock["identities"]
        self.assertEqual(ids["dino_factory_manifest_sha256"], DINO_FACTORY_MANIFEST_SHA256)
        for seed in ("S1", "S2", "S3"):
            self.assertEqual(
                ids[f"r07_{seed.lower()}_run_record_sha256"],
                R07_RUN_RECORDS[seed]["sha256"],
            )
        drifted = dict(lock)
        drifted["identities"] = dict(ids)
        drifted["identities"]["r07_s2_run_record_sha256"] = "0" * 64
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

    def test_execution_lock_forbids_postclosure_full_raw_rebuild(self):
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v2.json")
        validate_execution_lock(lock)
        drifted = dict(lock)
        drifted["historical_compare"] = dict(lock["historical_compare"])
        drifted["historical_compare"]["full_ext_i_route"] = dict(
            lock["historical_compare"]["full_ext_i_route"]
        )
        drifted["historical_compare"]["full_ext_i_route"]["raw_postclosure_rebuild_forbidden"] = False
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

    def test_seed_row_identity_must_be_exact(self):
        base = [{"stable_row_id": "a", "target_class_index": 1, "predicted_class_index": 1}]
        with self.assertRaises(TrackBError):
            validate_same_prediction_surface({
                "S1": base,
                "S2": [{"stable_row_id": "b", "target_class_index": 1, "predicted_class_index": 1}],
                "S3": base,
            })

    def test_representative_is_smallest_raw_sha_not_path_id(self):
        r1 = ImageAuditRecord("z", "a.jpg", "x", 1, "f" * 64, 0, 0, 10, 10)
        r2 = ImageAuditRecord("a", "b.jpg", "x", 1, "0" * 64, 0, 0, 10, 10)
        rows = representative_manifest([r1, r2], [["z", "a"]])
        self.assertEqual(rows[0]["representative_row_id"], "a")
        self.assertEqual(rows[0]["representative_raw_sha256"], "0" * 64)

    def test_bootstrap_is_reproducible_and_shared_across_seeds(self):
        rows = []
        for i in range(12):
            target = 1 if i < 6 else 2
            rows.append({"stable_row_id": str(i), "target_class_index": target, "predicted_class_index": target})
        seed_rows = {
            "S1": [dict(r) for r in rows],
            "S2": [dict(r) for r in rows],
            "S3": [dict(r) for r in rows],
        }
        a = bootstrap_three_seed_macro_f1(seed_rows, [1, 2], replicates=100, seed=409883112, chunk_replicates=20)
        b = bootstrap_three_seed_macro_f1(seed_rows, [1, 2], replicates=100, seed=409883112, chunk_replicates=20)
        self.assertEqual(a, b)
        self.assertEqual(a["three_seed"]["bootstrap_mean_macro_f1_ci95_percentile"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
