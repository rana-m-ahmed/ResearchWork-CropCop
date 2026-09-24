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
    audit_policy_from_lock,
    assign_candidate_grade,
    build_candidate_seal,
    mapped_scope_metrics,
    validate_prior_attempt_for_rerun,
    validate_same_prediction_surface,
    verify_candidate_seal,
    git_blob_sha1,
    verify_code_attestation,
    load_json,
    validate_execution_lock,
)
from cropcop_je.hashing import sha256_json
from cropcop_je.trackb_r07_analysis import bootstrap_three_seed_macro_f1
from cropcop_je.trackb_r07_audit import ImageAuditRecord, deterministic_representative_order, representative_manifest, topk_cosine_neighbors
from cropcop_je.trackb_r07_ops import _checksum_matches, _classify_github_push_failure, _parse_gvlid_checksum_authority, load_kaggle_secret


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
                "attestation_id": "TRACKB_CODE_ATTESTATION_v4",
                "parent_track_a_closure_commit": "604aafd51e20e70098ce4af647e90c8ff558a9e8",
                "files": [{"path": "x.py", "git_blob_sha1": git_blob_sha1(f)}],
            }
            ap = root / "att.json"
            ap.write_text(json.dumps(att), encoding="utf-8")
            verify_code_attestation(root, ap)
            f.write_text("print('drift')\n", encoding="utf-8")
            with self.assertRaises(TrackBError):
                verify_code_attestation(root, ap)

    def test_v3_candidate_roles_preserve_frozen_v2_selection(self):
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

    def test_v4_lock_requires_hardened_orchestration_boundaries(self):
        lock = load_json(
            ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v4.json"
        )
        validate_execution_lock(lock)

        orchestration = lock["orchestration"]
        self.assertEqual(
            orchestration["operator_notebook_policy"],
            "TWO_SUPPORTED_NOTEBOOKS_ONLY",
        )
        self.assertTrue(orchestration["attached_code_authenticated_before_execution"])
        self.assertTrue(orchestration["full_role_content_binding_required"])
        self.assertTrue(orchestration["content_addressed_private_datasets"])
        self.assertTrue(orchestration["immutable_qualification_bundle_required"])
        self.assertFalse(orchestration["claim_recomputes_qualification"])
        self.assertTrue(orchestration["claim_single_writer_lease_required"])
        self.assertEqual(orchestration["qualification_protected_prediction_count"], 0)
        self.assertFalse(orchestration["qualification_requires_github_token"])
        self.assertFalse(orchestration["claim_requires_github_token"])
        self.assertTrue(orchestration["v1_test_reopen_forbidden"])
        self.assertEqual(
            orchestration["durable_attempt_states"],
            [
                "PROTECTED_INFERENCE_STARTED",
                "SCIENCE_QA_PASS",
                "PRIVATE_ARCHIVE_VERIFIED",
                "TRACK_B_CLOSED",
            ],
        )

        source_integrity = lock["source_integrity"]
        self.assertTrue(source_integrity["gvlid_pinned_official_companion_ledger_required"])
        self.assertTrue(source_integrity["irish_potato_official_archive_checksum_required"])
        self.assertTrue(source_integrity["external_source_manifest_toc_tou_binding_required"])
        self.assertTrue(source_integrity["full_attached_role_content_binding_required"])

        runtime = lock["runtime_policy"]
        self.assertTrue(runtime["torch_deterministic_algorithms_required"])
        self.assertTrue(runtime["opencv_ransac_pair_seeded"])
        self.assertTrue(runtime["dino_topk_cutoff_ties_stable_by_reference_index"])
        self.assertEqual(runtime["max_audit_candidate_pairs_per_surface"], 5000000)

        self.assertEqual(lock["kaggle"]["target_accelerator"], "T4x2")
        self.assertEqual(lock["kaggle"]["publication_reserve_minutes"], 90)
        self.assertFalse(lock["kaggle"]["protected_inference_network_dependency"])
        self.assertFalse(lock["closure_policy"]["qualification_recomputed_during_claim"])

        import copy
        drifted = copy.deepcopy(lock)
        drifted["orchestration"]["claim_recomputes_qualification"] = True
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

        drifted = copy.deepcopy(lock)
        drifted["runtime_policy"]["opencv_ransac_pair_seeded"] = False
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

        drifted = copy.deepcopy(lock)
        drifted["source_integrity"]["full_attached_role_content_binding_required"] = False
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

    def test_github_push_failure_classifier_distinguishes_bad_token(self):
        message = _classify_github_push_failure(
            "remote: Invalid username or token. Password authentication is not supported."
        )
        self.assertIn("invalid", message.lower())
        self.assertIn("CROPCOP_GITHUB_TOKEN", message)

    def test_github_push_failure_classifier_distinguishes_permission_scope(self):
        message = _classify_github_push_failure(
            "remote: Write access to repository not granted. fatal: unable to access: 403"
        )
        self.assertIn("lacks repository write permission", message)
        self.assertIn("Contents: Read and write", message)

    def test_secret_loader_prefers_environment_for_fresh_subprocess_runtime(self):
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
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v4.json")
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
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v4.json")
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

    def test_audit_policy_is_executable_lock_source_of_truth(self):
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v4.json")
        policy = audit_policy_from_lock(lock)
        self.assertEqual(policy.phash_radius, 12)
        self.assertEqual(policy.dhash_radius, 10)
        self.assertEqual(policy.dino_top_k, 50)
        self.assertEqual(policy.orb_max_side, 800)
        self.assertEqual(policy.orb_nfeatures, 1200)
        self.assertEqual(policy.bootstrap_replicates, 5000)
        self.assertEqual(policy.bootstrap_seed, 409883112)
        self.assertEqual(policy.external_family_order_seed, 1936263114)

    def test_lock_rejects_operational_threshold_drift(self):
        import copy
        lock = load_json(ROOT / "journal_extension" / "track_b_r07" / "TRACKB_R07_EXECUTION_LOCK_v4.json")
        mutations = [
            ("candidate_generation", "dino_top_k", 49),
            ("geometric_acceptance", "lowe_ratio", 0.74),
            ("geometric_acceptance", "orb_features_max", 1199),
            ("bootstrap", "replicates", 4999),
        ]
        for section, key, value in mutations:
            drifted = copy.deepcopy(lock)
            drifted[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(TrackBError):
                    validate_execution_lock(drifted)
        drifted = copy.deepcopy(lock)
        drifted["candidate_generation"]["phash"]["hamming_max"] = 11
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)
        drifted = copy.deepcopy(lock)
        drifted["candidate_generation"]["dhash"]["hamming_max"] = 9
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)
        drifted = copy.deepcopy(lock)
        drifted["external_family_order_seed"] = 1
        with self.assertRaises(TrackBError):
            validate_execution_lock(drifted)

    def test_topk_cosine_neighbors_matches_reference_on_random_surface(self):
        import numpy as np
        import torch

        rng = np.random.default_rng(1701)
        q = rng.normal(size=(37, 23)).astype(np.float32)
        r = rng.normal(size=(113, 23)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        r /= np.linalg.norm(r, axis=1, keepdims=True)

        observed_idx, observed_score = topk_cosine_neighbors(
            q, r, k=11, device="cpu", block_rows=8
        )

        score = torch.from_numpy(q) @ torch.from_numpy(r).T
        expected_idx = []
        expected_score = []
        for row in score:
            values, _indices = torch.topk(row, k=11, largest=True, sorted=True)
            threshold = values[-1]
            strict_idx = torch.nonzero(row > threshold, as_tuple=False).flatten()
            tie_idx = torch.nonzero(row == threshold, as_tuple=False).flatten()
            slots = 11 - int(strict_idx.numel())
            chosen = torch.cat((strict_idx, torch.sort(tie_idx).values[:slots]))
            chosen_scores = row[chosen]
            order = torch.argsort(chosen_scores, descending=True, stable=True)
            expected_idx.append(chosen[order].numpy())
            expected_score.append(chosen_scores[order].numpy())

        np.testing.assert_array_equal(observed_idx, np.stack(expected_idx))
        np.testing.assert_allclose(
            observed_score, np.stack(expected_score), rtol=0.0, atol=0.0
        )

    def test_topk_cosine_neighbors_matches_reference_on_cutoff_ties(self):
        import numpy as np
        import torch

        q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        r = np.asarray([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        observed_idx, observed_score = topk_cosine_neighbors(
            q, r, k=2, device="cpu", block_rows=2
        )

        score = torch.from_numpy(q) @ torch.from_numpy(r).T
        expected_idx = []
        expected_score = []
        for row in score:
            values, _indices = torch.topk(row, k=2, largest=True, sorted=True)
            threshold = values[-1]
            strict_idx = torch.nonzero(row > threshold, as_tuple=False).flatten()
            tie_idx = torch.nonzero(row == threshold, as_tuple=False).flatten()
            slots = 2 - int(strict_idx.numel())
            chosen = torch.cat((strict_idx, torch.sort(tie_idx).values[:slots]))
            chosen_scores = row[chosen]
            order = torch.argsort(chosen_scores, descending=True, stable=True)
            expected_idx.append(chosen[order].numpy())
            expected_score.append(chosen_scores[order].numpy())

        np.testing.assert_array_equal(observed_idx, np.stack(expected_idx))
        np.testing.assert_allclose(
            observed_score, np.stack(expected_score), rtol=0.0, atol=0.0
        )

    def test_operator_sources_fail_closed_to_qualification(self):
        master = (ROOT / "journal_extension" / "scripts" / "run_trackb_r07_master.py").read_text(encoding="utf-8")
        runner = (ROOT / "journal_extension" / "scripts" / "run_trackb_r07.py").read_text(encoding="utf-8")
        notebook = (ROOT / "journal_extension" / "kaggle" / "trackb_r07_master.ipynb").read_text(encoding="utf-8")
        self.assertIn('choices=["qualification", "claim"], default="qualification"', master)
        self.assertIn('PASS_TRACKB_PREINFERENCE_QUALIFICATION', master)
        self.assertIn('--authorized-qualification-science-sha256', master)
        self.assertIn('claim mode requires a reviewed --authorized-qualification-science-sha256', master)
        self.assertIn('--authorized-qualification-science-sha256', runner)
        self.assertIn('authorized_qualification_science', runner)
        self.assertIn('current_qualification_science', runner)
        self.assertIn('TRACKB_QUALIFICATION_AUTHORIZATION.json', runner)
        self.assertIn('validate_trackb_preinference_qualification.py', master)
        self.assertIn('choices=["preflight", "qualification", "claim", "all"]', runner)
        self.assertIn('--qualification-root', runner)
        self.assertIn('claim mode requires --qualification-root', runner)
        self.assertIn('qualification_recomputed": False', runner)
        self.assertLess(
            runner.index('if args.mode == "claim":'),
            runner.index('stage("1-4 :: prediction-blind source verification'),
        )
        self.assertIn("--execution-mode", notebook)
        self.assertIn("qualification", notebook)
        self.assertIn("TRACKB_PREINFERENCE_QA.json", notebook)
        self.assertNotIn("CROPCOP_GITHUB_TOKEN", notebook)
        self.assertNotIn("verify_github_repository_push_access", notebook)
        self.assertNotIn("PASS_AUTOMATED_TRACK_B_COMPLETE", notebook)

    def test_prediction_blind_science_identity_excludes_execution_metadata(self):
        source = (
            ROOT / "journal_extension" / "scripts" / "run_trackb_r07.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _prediction_blind_science_manifest(")
        end = source.index("def _stable_science_manifest(", start)
        helper = source[start:end]
        for forbidden in (
            "sealed_at_utc",
            "retrieved_at",
            "final_qa_sha256",
            "candidate_input_manifest_sha256",
            "source_metadata_record_sha256",
        ):
            self.assertNotIn(forbidden, helper)
        self.assertIn("qualification_science_sha256", helper)
        self.assertIn("TRACKB_PREDICTION_BLIND_SCIENCE.json", helper)

    def test_preinference_validator_forbids_claim_artifacts(self):
        source = (
            ROOT / "journal_extension" / "scripts" / "validate_trackb_preinference_qualification.py"
        ).read_text(encoding="utf-8")
        for token in (
            "TRACKB_FINAL_CLOSURE.json",
            "TRACKB_FINAL_QA.json",
            "TRACKB_ATTEMPT_STATE.json",
            "protected_external_prediction_count",
            "PASS_INDEPENDENT_PREINFERENCE_QA",
            "verify_candidate_seal",
            "TRACKB_PREDICTION_BLIND_SCIENCE.json",
            "qualification_science_sha256",
            "reconstructed_science",
            "qualification_science_sha256",
            "TRACKB_PREDICTION_BLIND_SCIENCE.json",
        ):
            self.assertIn(token, source)

    def test_seeded_representative_order_is_traversal_invariant(self):
        rows = [
            {"representative_raw_sha256": f"{i:064x}", "family_id": f"F{i}", "representative_row_id": f"R{i}"}
            for i in range(12)
        ]
        a = deterministic_representative_order(rows, seed=1936263114)
        b = deterministic_representative_order(list(reversed(rows)), seed=1936263114)
        self.assertEqual(a, b)
        self.assertEqual(
            {row["representative_row_id"] for row in a},
            {row["representative_row_id"] for row in rows},
        )
        c = deterministic_representative_order(rows, seed=1936263115)
        self.assertNotEqual(
            [row["representative_row_id"] for row in a],
            [row["representative_row_id"] for row in c],
        )

    def test_checksum_verifier_supports_zenodo_md5_and_sha256(self):
        import hashlib, tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.bin"
            path.write_bytes(b"track-b-source")
            md5 = hashlib.md5(path.read_bytes()).hexdigest()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(_checksum_matches(path, f"md5:{md5}"))
            self.assertTrue(_checksum_matches(path, f"sha256:{sha}"))
            self.assertFalse(_checksum_matches(path, "md5:" + "0" * 32))

    def test_gvlid_checksum_authority_parser_is_deterministic(self):
        path = (
            ROOT
            / "journal_extension"
            / "track_b_r07"
            / "external_authority"
            / "gvlid_v5_checksums.csv"
        )
        first = _parse_gvlid_checksum_authority(path)
        second = _parse_gvlid_checksum_authority(path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3477)

    def test_attempt_rerun_gate_allows_only_unchanged_interrupted_attempt(self):
        digest = "a" * 64
        gate = validate_prior_attempt_for_rerun(
            {
                "attempt_id": "old",
                "protected_inference_ever": True,
                "science_preimage_sha256": digest,
                "status": "PROTECTED_INFERENCE_STARTED",
            },
            current_science_preimage_sha256=digest,
        )
        self.assertTrue(gate["prior_attempt_with_protected_inference"])
        self.assertEqual(gate["parent_attempt_id"], "old")

        with self.assertRaises(TrackBError):
            validate_prior_attempt_for_rerun(
                {
                    "attempt_id": "old",
                    "protected_inference_ever": True,
                    "science_preimage_sha256": "b" * 64,
                    "status": "PROTECTED_INFERENCE_STARTED",
                },
                current_science_preimage_sha256=digest,
            )

        for status in ("SCIENCE_QA_PASS", "PRIVATE_ARCHIVE_VERIFIED", "PUBLICATION_COMPLETE"):
            with self.subTest(status=status):
                with self.assertRaises(TrackBError):
                    validate_prior_attempt_for_rerun(
                        {
                            "attempt_id": "old",
                            "protected_inference_ever": True,
                            "science_preimage_sha256": digest,
                            "status": status,
                        },
                        current_science_preimage_sha256=digest,
                    )

    def test_v5_deterministic_orb_ransac_contract_is_source_locked(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_audit.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_CV2_RANSAC_LOCK", source)
        self.assertIn("with _CV2_RANSAC_LOCK:", source)
        self.assertIn("cv2.setRNGSeed(int(rng_seed) & 0x7FFFFFFF)", source)
        self.assertIn("cv2.findHomography(", source)

    def test_v5_time_budget_guards_cover_qualification_claim_and_packaging(self):
        source = (
            ROOT / "journal_extension" / "scripts" / "run_trackb_r07.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _require_remaining_time(", source)
        self.assertIn('required_seconds=5 * 3600', source)
        self.assertIn('stage_name="prediction-blind qualification"', source)
        self.assertIn('required_seconds=2 * 3600', source)
        self.assertIn('stage_name="protected claim"', source)
        self.assertIn('required_seconds=45 * 60', source)
        self.assertIn('stage_name="claim evidence packaging"', source)

    def test_v5_capacity_preflight_enforces_t4x2_vram_and_scratch(self):
        source = (
            ROOT / "journal_extension" / "scripts" / "run_trackb_r07.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _preflight_kaggle_capacity(", source)
        self.assertIn("device_count < 2", source)
        self.assertIn('"T4" not in name.upper()', source)
        self.assertIn("min_total_vram = 12 * 1024**3", source)
        self.assertIn("min_free_vram = 8 * 1024**3", source)
        self.assertIn("shutil.disk_usage(scratch_root).free", source)
        self.assertIn("expected_external_images=0 if args.mode == \"claim\" else 3477 + 58709", source)

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
