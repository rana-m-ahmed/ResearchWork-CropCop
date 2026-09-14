from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SCIENCE_SHA = "9a72e9466a9a3e7429e0e36a028edac662f83146"
RUNTIME_SHA = "280743cf619d47b093944de6ea63be62dcb8f7ae"
RUNTIME_BRANCH = "ops-tracka-kaggle-master-runtime-v6-280743c"
LAUNCHER = "master_launch_guard_v6.py"
DRIVER = "master_account_driver_v6.py"
AUTHORITY = "EAAI-JE-SDL-v2.1-QA"
SEALER_BLOB = "90a918fcdf14130b44b9c4ec24b0b60b1706bb2c"
NOTEBOOKS = {
    "TRACKA_V12_MASTER_K1.ipynb": "K1",
    "TRACKA_V12_MASTER_K2.ipynb": "K2",
    "TRACKA_V12_MASTER_K3.ipynb": "K3",
}


def text(path: str) -> str:
    return (OPS / path).read_text(encoding="utf-8")


class TrackAV12KaggleOperatorDistributionTests(unittest.TestCase):
    def test_operator_contract_is_v6_and_exact(self):
        contract = json.loads(text("OPERATOR_CONTRACT.json"))
        self.assertEqual(contract["schema_version"], "3.6")
        self.assertEqual(contract["status"], "LOCKED_MASTER_OPERATOR_CONTRACT")
        self.assertEqual(contract["scientific_source_sha"], SCIENCE_SHA)
        self.assertEqual(contract["operator_runtime_sha"], RUNTIME_SHA)
        self.assertEqual(contract["operator_runtime_branch"], RUNTIME_BRANCH)
        self.assertEqual(contract["launcher"], LAUNCHER)
        self.assertEqual(contract["driver"], DRIVER)
        self.assertEqual(contract["authority_id"], AUTHORITY)
        self.assertEqual(contract["notebooks"], list(NOTEBOOKS))

        dataset = contract["dataset"]
        self.assertFalse(dataset["preferred_paths_are_trusted_without_verification"])
        self.assertTrue(dataset["fail_fast_before_dependency_install"])
        self.assertTrue(dataset["nested_split_root_discovery"])
        self.assertEqual(dataset["root_qualification_surfaces"], ["train", "val"])
        self.assertFalse(dataset["protected_test_opened_for_root_qualification"])
        self.assertTrue(dataset["resolved_paths_bound_for_downstream_reuse"])

        tv = contract["torchvision_provenance"]
        self.assertFalse(tv["download_receipt_is_scientific_provenance"])
        self.assertTrue(tv["exact_tensor_match_required"])
        self.assertTrue(tv["candidate_and_official_tensor_identity_must_match"])
        self.assertTrue(tv["frozen_validator_required_before_g1a_seal"])

        compat = contract["g1a_sealer_compatibility"]
        self.assertFalse(compat["frozen_sealer_modified"])
        self.assertEqual(compat["required_git_blob"], SEALER_BLOB)
        self.assertEqual(compat["verified_defect"], "TORCHVISION_VERSION referenced but not bound")
        self.assertEqual(compat["compatibility_runner"], "master_g1a_sealer_compat_v6.py")
        self.assertEqual(compat["injected_global"], "TORCHVISION_VERSION")
        self.assertEqual(compat["injected_value_source"], "cropcop_je.secondary.TORCHVISION_VERSION")
        self.assertEqual(compat["required_injected_value"], "0.27.1")
        self.assertTrue(compat["ast_shape_check_required"])
        self.assertTrue(compat["unexpected_sealer_drift_fails_closed"])

        kaggle = contract["kaggle"]
        self.assertEqual(kaggle["g2a_distinct_private_dataset_count"], 5)
        self.assertEqual(kaggle["scientific_distinct_private_dataset_count"], 11)
        self.assertEqual(kaggle["canonical_g1a_private_dataset_count"], 1)
        self.assertTrue(kaggle["optional_expected_username_gate"])
        self.assertTrue(kaggle["per_account_launch_lock"])
        self.assertFalse(kaggle["duplicate_launch_executes_second_master"])
        self.assertTrue(kaggle["duplicate_launch_follows_owner_terminal_status"])
        self.assertTrue(kaggle["owner_release_without_valid_status_allows_safe_takeover"])

        publication = contract["publication"]
        self.assertTrue(publication["write_preflight_before_stack_repair"])
        self.assertTrue(publication["byte_identical_remote_is_successful_noop"])
        self.assertTrue(publication["partial_bundle_incremental_publication"])
        self.assertTrue(publication["changed_file_only_update"])
        self.assertTrue(publication["final_full_bundle_roundtrip_required"])
        self.assertTrue(publication["remote_branch_must_descend_from_science_sha"])
        self.assertTrue(publication["post_error_remote_roundtrip_recovery"])
        self.assertFalse(publication["frozen_scientific_runner_git_publication_enabled"])
        self.assertTrue(publication["scientific_evidence_published_by_master_parent"])
        self.assertFalse(publication["publication_failure_invalidates_science"])

        self.assertTrue(contract["pre_science_io"]["science_checkout_bounded_retry"])
        self.assertTrue(contract["pre_science_io"]["torchvision_upstream_bounded_retry"])
        self.assertTrue(contract["pre_science_io"]["r13_upstream_bounded_retry"])
        self.assertFalse(contract["pre_science_io"]["scientific_result_steps_auto_retried"])
        self.assertFalse(contract["rules"]["scientific_source_may_follow_operator_head"])
        self.assertFalse(contract["rules"]["notebooks_may_follow_distribution_head"])
        self.assertFalse(contract["rules"]["protected_test_open_before_track_a_closure"])
        self.assertFalse(contract["rules"]["external_prediction_open_before_track_a_closure"])
        self.assertTrue(contract["rules"]["science_requires_durability_bound_go"])
        self.assertFalse(contract["rules"]["ddp_allowed"])
        self.assertFalse(contract["rules"]["dataparallel_allowed"])
        self.assertFalse(contract["rules"]["fsdp_allowed"])

    def test_exactly_three_master_notebooks_remain(self):
        observed = sorted(path.name for path in OPS.glob("*.ipynb"))
        self.assertEqual(observed, sorted(NOTEBOOKS))
        self.assertFalse(any(name[:2].isdigit() for name in observed))

    def test_master_notebooks_are_clean_pinned_and_launcher_only(self):
        for name, account in NOTEBOOKS.items():
            with self.subTest(name=name):
                nb = json.loads(text(name))
                self.assertEqual(nb["nbformat"], 4)
                self.assertEqual(nb["nbformat_minor"], 5)
                meta = nb["metadata"]["cropcop_operator"]
                self.assertEqual(meta["schema_version"], "3.6")
                self.assertEqual(meta["account_id"], account)
                self.assertEqual(meta["science_sha"], SCIENCE_SHA)
                self.assertEqual(meta["operator_runtime_sha"], RUNTIME_SHA)
                self.assertEqual(meta["operator_runtime_branch"], RUNTIME_BRANCH)
                self.assertEqual(meta["launcher"], LAUNCHER)
                self.assertEqual(meta["driver"], DRIVER)
                self.assertEqual(meta["authority"], AUTHORITY)
                body = "\n".join("".join(c.get("source") or []) for c in nb["cells"])
                self.assertIn(RUNTIME_SHA, body)
                self.assertIn(RUNTIME_BRANCH, body)
                self.assertIn(LAUNCHER, body)
                self.assertNotIn(DRIVER, body)
                self.assertIn(f"str(launcher), '{account}'", body)
                self.assertIn("CropCop_Final_v1", body)
                self.assertIn("V1 / 'audit' / 'final_manifest.csv'", body)
                self.assertIn("V1 / 'audit' / 'class_to_idx.json'", body)
                self.assertIn("V1 / 'dataset'", body)
                for cell in nb["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])
                        compile("".join(cell.get("source") or []), f"{name}:cell", "exec")

    def test_master_notebooks_do_not_embed_secret_values(self):
        for name in NOTEBOOKS:
            body = text(name)
            self.assertNotIn("github_pat_", body)
            self.assertNotIn("ghp_", body)
            self.assertNotIn("KAGGLE_KEY =", body)
            self.assertNotIn("CROPCOP_GITHUB_TOKEN =", body)

    def test_k1_is_only_notebook_requiring_principal_g1(self):
        self.assertIn("CROPCOP_PRINCIPAL_G1", text("TRACKA_V12_MASTER_K1.ipynb"))
        self.assertIn("cropcop-g1-sealed/G1_PACKAGE", text("TRACKA_V12_MASTER_K1.ipynb"))
        self.assertNotIn("CROPCOP_PRINCIPAL_G1", text("TRACKA_V12_MASTER_K2.ipynb"))
        self.assertNotIn("CROPCOP_PRINCIPAL_G1", text("TRACKA_V12_MASTER_K3.ipynb"))

    def test_distribution_documents_same_runtime(self):
        distribution = json.loads(text("MASTER_NOTEBOOK_DISTRIBUTION.json"))
        freeze = json.loads(text("MASTER_RUNTIME_FREEZE.json"))
        marker = text(".runtime-freeze-marker")
        self.assertEqual(distribution["schema_version"], "3.6")
        self.assertEqual(distribution["operator_runtime_sha"], RUNTIME_SHA)
        self.assertEqual(distribution["operator_runtime_branch"], RUNTIME_BRANCH)
        self.assertEqual(distribution["launcher"], LAUNCHER)
        self.assertEqual(distribution["driver"], DRIVER)
        self.assertTrue(distribution["preferred_real_kaggle_mount_paths_embedded"])
        self.assertTrue(distribution["preferred_paths_still_hash_structure_verified"])
        self.assertTrue(distribution["two_stage_torchvision_pretrained_provenance"])
        self.assertTrue(distribution["frozen_torchvision_provenance_validator_before_g1a"])
        self.assertFalse(distribution["frozen_g1a_sealer_source_modified"])
        self.assertEqual(distribution["frozen_g1a_sealer_git_blob"], SEALER_BLOB)
        self.assertTrue(distribution["g1a_sealer_missing_global_compatibility_guard"])
        self.assertTrue(distribution["duplicate_account_launch_serialization"])
        self.assertTrue(distribution["duplicate_launch_follows_canonical_owner"])
        self.assertTrue(distribution["partial_bundle_incremental_publication"])
        self.assertTrue(distribution["final_full_bundle_roundtrip_required"])
        self.assertEqual(freeze["runtime_candidate_sha"], RUNTIME_SHA)
        self.assertEqual(freeze["runtime_branch"], RUNTIME_BRANCH)
        self.assertIn(RUNTIME_SHA, marker)
        self.assertIn(RUNTIME_BRANCH, marker)

    def test_protected_surfaces_remain_closed_in_master_runtime(self):
        combined = "\n".join(text(name) for name in (
            "master_account_driver_v6.py", "master_g1a_v6.py", "master_g2a.py", "master_control.py", "master_science_v4.py"
        ))
        self.assertNotIn("DS-V1-TEST-CONSUMED", combined)
        self.assertNotIn("--test", combined)
        self.assertNotIn("track_b", combined.lower())
        self.assertNotIn("track_c", combined.lower())


if __name__ == "__main__":
    unittest.main()
