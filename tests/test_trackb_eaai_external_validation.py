from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.trackb_eaai_common import (  # noqa: E402
    CHECKPOINTS,
    Bundle,
    TrackBEAAIError,
    _contains_forbidden_test_surface,
    load_protocol,
    sha256_file,
    validate_materialization_pair,
)
from cropcop_je.trackb_eaai_eval import (  # noqa: E402
    aggregate_three_seed,
    metrics,
    paired_stratified_bootstrap,
)


class TrackBEAAISimplifiedTests(unittest.TestCase):
    def setUp(self):
        self.protocol_path = (
            ROOT
            / "journal_extension"
            / "track_b_r07"
            / "TRACKB_EAAI_PROTOCOL_v1.json"
        )
        self.runner_path = (
            ROOT
            / "journal_extension"
            / "scripts"
            / "run_trackb_eaai_external_validation.py"
        )

    def test_protocol_is_frozen_pre_results_and_seven_class(self):
        protocol = load_protocol(self.protocol_path)
        self.assertEqual(
            protocol["protocol_id"],
            "TRACKB_EAAI_EXTERNAL_VALIDATION_v1",
        )
        self.assertEqual(protocol["status"], "FROZEN_PRE_RESULTS")
        self.assertFalse(
            protocol["amendment"][
                "protected_external_predictions_observed_before_amendment"
            ]
        )
        self.assertFalse(
            protocol["amendment"][
                "external_metric_results_used_to_design_amendment"
            ]
        )
        self.assertEqual(protocol["checkpoints"], CHECKPOINTS)
        mappings = [
            mapping
            for spec in protocol["datasets"].values()
            for mapping in spec["mapping"].values()
        ]
        self.assertEqual(len(mappings), 7)
        self.assertEqual(len(set(mappings)), 7)
        self.assertEqual(
            protocol["leakage_surface"]["image_count"],
            92744,
        )
        self.assertFalse(
            protocol["leakage_surface"]["v1_test_accessed"]
        )

    def test_prediction_policy_is_native_120_and_no_renormalization(self):
        protocol = load_protocol(self.protocol_path)
        policy = protocol["prediction_policy"]
        self.assertEqual(policy["native_output_classes"], 120)
        self.assertFalse(policy["mapped_subset_logit_renormalization"])
        self.assertTrue(policy["predictions_outside_mapped_scope_are_errors"])
        self.assertFalse(policy["external_retraining"])
        self.assertFalse(policy["external_finetuning"])
        self.assertFalse(policy["external_calibration"])
        self.assertTrue(policy["best_seed_selection_forbidden"])

    def test_structural_v1_guard_allows_false_provenance(self):
        safe = {
            "role": "historical_compare",
            "coverage_scope": "V1_TRAIN_VAL_ONLY",
            "v1_test_image_bytes_accessed": False,
            "files": {
                "manifest": {
                    "path": "historical_manifest.csv",
                    "sha256": "a" * 64,
                }
            },
        }
        self.assertFalse(_contains_forbidden_test_surface(safe))

    def test_structural_v1_guard_rejects_true_access_or_test_path(self):
        self.assertTrue(
            _contains_forbidden_test_surface(
                {"v1_test_image_bytes_accessed": True}
            )
        )
        self.assertTrue(
            _contains_forbidden_test_surface(
                {"data_root": "safe/v1_test/images"}
            )
        )
        self.assertTrue(
            _contains_forbidden_test_surface(
                {"surface": "DS-V1-TEST-CONSUMED"}
            )
        )

    def test_out_of_scope_prediction_is_error_not_renormalized(self):
        rows = [
            {
                "row_id": "a",
                "target_class_index": 1,
                "predicted_class_index": 1,
            },
            {
                "row_id": "b",
                "target_class_index": 1,
                "predicted_class_index": 99,
            },
            {
                "row_id": "c",
                "target_class_index": 2,
                "predicted_class_index": 2,
            },
            {
                "row_id": "d",
                "target_class_index": 2,
                "predicted_class_index": 1,
            },
        ]
        observed = metrics(rows, {1, 2})
        self.assertAlmostEqual(observed["accuracy"], 0.5)
        self.assertAlmostEqual(
            observed["out_of_mapped_scope_prediction_rate"],
            0.25,
        )
        self.assertEqual(
            observed["confusion_matrix_mapped_plus_oos"]["1"]["-1"],
            1,
        )

    def test_three_seed_aggregate_uses_sample_sd(self):
        seed_metrics = {
            "S1": {
                "macro_f1": 0.6,
                "accuracy": 0.7,
                "balanced_accuracy": 0.6,
                "out_of_mapped_scope_prediction_rate": 0.1,
            },
            "S2": {
                "macro_f1": 0.7,
                "accuracy": 0.8,
                "balanced_accuracy": 0.7,
                "out_of_mapped_scope_prediction_rate": 0.2,
            },
            "S3": {
                "macro_f1": 0.8,
                "accuracy": 0.9,
                "balanced_accuracy": 0.8,
                "out_of_mapped_scope_prediction_rate": 0.3,
            },
        }
        out = aggregate_three_seed(seed_metrics)
        self.assertAlmostEqual(out["macro_f1"]["mean"], 0.7)
        self.assertAlmostEqual(out["macro_f1"]["sample_sd"], 0.1)

    def test_paired_bootstrap_is_deterministic_and_seed_aligned(self):
        base = [
            ("a", 1, 1, 1, 1),
            ("b", 1, 1, 2, 1),
            ("c", 1, 2, 2, 2),
            ("d", 2, 2, 2, 2),
            ("e", 2, 2, 1, 2),
            ("f", 2, 1, 1, 1),
        ]
        seed_rows = {}
        for index, seed in enumerate(("S1", "S2", "S3"), start=2):
            seed_rows[seed] = [
                {
                    "row_id": row[0],
                    "target_class_index": row[1],
                    "predicted_class_index": row[index],
                }
                for row in base
            ]
        one = paired_stratified_bootstrap(
            seed_rows,
            labels={1, 2},
            replicates=250,
            rng_seed=409883112,
        )
        two = paired_stratified_bootstrap(
            seed_rows,
            labels={1, 2},
            replicates=250,
            rng_seed=409883112,
        )
        self.assertEqual(one, two)
        self.assertEqual(one["replicates"], 250)
        self.assertEqual(set(one["per_seed_macro_f1_95pct"]), {"S1", "S2", "S3"})

    def test_bootstrap_rejects_nonidentical_seed_surfaces(self):
        seed_rows = {
            "S1": [
                {
                    "row_id": "a",
                    "target_class_index": 1,
                    "predicted_class_index": 1,
                }
            ],
            "S2": [
                {
                    "row_id": "b",
                    "target_class_index": 1,
                    "predicted_class_index": 1,
                }
            ],
            "S3": [
                {
                    "row_id": "a",
                    "target_class_index": 1,
                    "predicted_class_index": 1,
                }
            ],
        }
        with self.assertRaises(TrackBEAAIError):
            paired_stratified_bootstrap(
                seed_rows,
                labels={1},
                replicates=10,
                rng_seed=1,
            )

    def test_new_runner_is_nonempty_and_has_no_legacy_heavy_path(self):
        source = self.runner_path.read_text(encoding="utf-8")
        self.assertGreater(len(source), 5000)
        for forbidden in (
            "topk_cosine_neighbors",
            "verify_orb_pair",
            "BFMatcher",
            "findHomography",
            "KAGGLE_API_TOKEN",
            "publish_private_kaggle_dataset",
            "claim_lease",
            "qualification_science_sha256",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("leakage_clean_primary", source)
        self.assertIn("mapped_scope", source.lower())

    def test_materialization_pair_accepts_matching_receipts_and_rejects_mixup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            roles = {}
            role_hashes = {}
            for role in ("core", "historical_compare", "gvlid_v5", "irish_potato"):
                role_root = root / role
                role_root.mkdir()
                manifest_path = role_root / "TRACKB_INPUT_MANIFEST.json"
                manifest = {"schema_version": "1.0", "role": role}
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                role_hashes[role] = sha256_file(manifest_path)
                roles[role] = Bundle(
                    role=role,
                    root=role_root.resolve(),
                    manifest_path=manifest_path.resolve(),
                    manifest=manifest,
                )

            materialization_id = "9" * 64
            common = {
                "schema_version": "2.0",
                "status": "PASS_PAIRED_TRACKB_INPUT_BUNDLE",
                "materialization_id": materialization_id,
                "repository_source_sha": "a" * 40,
                "scientific_execution_lock_sha256": "b" * 64,
                "scientific_code_attestation_sha256": "c" * 64,
                "external_source_lock_sha256": "d" * 64,
                "input_materialization_lock_sha256": "e" * 64,
                "role_manifest_sha256": role_hashes,
            }
            infra = {
                **common,
                "bundle_role": "TRACKB_INFRASTRUCTURE",
            }
            external = {
                **common,
                "bundle_role": "TRACKB_EXTERNAL",
            }
            infra_path = root / "TRACKB_INFRASTRUCTURE_BUNDLE.json"
            external_path = root / "TRACKB_EXTERNAL_BUNDLE.json"
            infra_path.write_text(
                json.dumps(infra, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            external_path.write_text(
                json.dumps(external, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            receipt = validate_materialization_pair(
                root,
                roles,
                materialization_id,
            )
            self.assertEqual(
                receipt["materialization_id"],
                materialization_id,
            )

            external["materialization_id"] = "8" * 64
            external_path.write_text(
                json.dumps(external, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TrackBEAAIError):
                validate_materialization_pair(
                    root,
                    roles,
                    materialization_id,
                )

    def test_kaggle_notebook_is_pinned_account_independent_and_single_run(self):
        notebook_path = (
            ROOT
            / "journal_extension"
            / "kaggle"
            / "TrackB_EAAI_External_Validation.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source") or [])
            for cell in notebook.get("cells") or []
        )
        self.assertIn(
            "d9e9a323834ae6d2d6b745e18e4023f8789a6c7e",
            source,
        )
        self.assertIn(
            "run_trackb_eaai_external_validation.py",
            source,
        )
        self.assertIn("'--mode', 'all'", source)
        self.assertIn("'--device', 'cuda:0'", source)
        self.assertNotIn("KAGGLE_API_TOKEN", source)
        self.assertNotIn("KAGGLE_OWNER", source)
        self.assertNotIn("ranaabdulrehmannn", source)
        self.assertNotIn("ranamuhammadahmed6", source)
        self.assertNotIn("qualification_science_sha256", source)
        self.assertNotIn("claim", source.lower())
        self.assertNotIn("BFMatcher", source)
        self.assertNotIn("topk_cosine_neighbors", source)

    def test_protocol_performance_never_controls_execution_pass(self):
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.assertTrue(
            protocol["stop_conditions"][
                "scientific_performance_never_causes_execution_failure"
            ]
        )


if __name__ == "__main__":
    unittest.main()
