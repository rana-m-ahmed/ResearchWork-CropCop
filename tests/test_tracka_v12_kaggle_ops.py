from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SCIENCE_SHA = "9a72e9466a9a3e7429e0e36a028edac662f83146"
RUNTIME_SHA = "208656f895af5c218a0998b582f2adbb81167eaa"
RUNTIME_BRANCH = "ops-tracka-kaggle-master-runtime-v6-208656f"
DRIVER = "master_account_driver_v6.py"
AUTHORITY = "EAAI-JE-SDL-v2.1-QA"
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
        self.assertEqual(contract["driver"], DRIVER)
        self.assertEqual(contract["authority_id"], AUTHORITY)
        self.assertEqual(contract["notebooks"], list(NOTEBOOKS))
        self.assertEqual(contract["kaggle"]["g2a_distinct_private_dataset_count"], 5)
        self.assertEqual(contract["kaggle"]["scientific_distinct_private_dataset_count"], 11)
        self.assertEqual(contract["kaggle"]["canonical_g1a_private_dataset_count"], 1)
        self.assertFalse(contract["dataset"]["preferred_paths_are_trusted_without_verification"])
        self.assertTrue(contract["dataset"]["fail_fast_before_dependency_install"])
        self.assertEqual(contract["dataset"]["root_qualification_surfaces"], ["train", "val"])
        self.assertFalse(contract["dataset"]["protected_test_opened_for_root_qualification"])
        compat = contract["frozen_g1a_sealer_compatibility"]
        self.assertFalse(compat["frozen_source_modified"])
        self.assertEqual(compat["static_unresolved_globals"], ["TORCHVISION_VERSION"])
        self.assertEqual(compat["expected_value"], "0.27.1")
        self.assertTrue(compat["real_subprocess_injection_tested"])
        concurrency = contract["concurrency"]
        self.assertTrue(concurrency["per_account_master_singleton"])
        self.assertFalse(concurrency["duplicate_invocation_enters_critical_section"])
        self.assertTrue(concurrency["duplicate_invocation_waits_and_mirrors_primary"])
        self.assertTrue(concurrency["stale_running_marker_fails_closed"])
        self.assertTrue(concurrency["notebook_runtime_checkout_process_isolated"])
        publication = contract["publication"]
        self.assertTrue(publication["byte_identical_remote_is_successful_noop"])
        self.assertTrue(publication["partial_bundle_incremental_publication"])
        self.assertTrue(publication["final_full_bundle_roundtrip_required"])
        self.assertFalse(publication["frozen_scientific_runner_git_publication_enabled"])
        self.assertFalse(contract["pre_science_io"]["scientific_result_steps_auto_retried"])
        for key in ("clean_after_checkout", "clean_after_stack_repair", "clean_after_g1a", "clean_after_g2a", "clean_after_control_plane", "clean_before_science"):
            self.assertTrue(contract["source_integrity"][key])
        self.assertFalse(contract["rules"]["protected_test_open_before_track_a_closure"])
        self.assertFalse(contract["rules"]["external_prediction_open_before_track_a_closure"])
        self.assertFalse(contract["rules"]["ddp_allowed"])
        self.assertFalse(contract["rules"]["dataparallel_allowed"])
        self.assertFalse(contract["rules"]["fsdp_allowed"])

    def test_exactly_three_master_notebooks_remain(self):
        observed = sorted(path.name for path in OPS.glob("*.ipynb"))
        self.assertEqual(observed, sorted(NOTEBOOKS))

    def test_master_notebooks_are_clean_pinned_process_isolated_and_account_specific(self):
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
                self.assertEqual(meta["driver"], DRIVER)
                self.assertEqual(meta["authority"], AUTHORITY)
                body = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
                for required in (
                    RUNTIME_SHA, RUNTIME_BRANCH, DRIVER, f"str(driver), '{account}'", "CropCop_Final_v1",
                    "V1 / 'audit' / 'final_manifest.csv'", "V1 / 'audit' / 'class_to_idx.json'", "V1 / 'dataset'",
                    "os.getpid()", "uuid.uuid4()", "Process-isolated runtime checkout",
                ):
                    self.assertIn(required, body)
                self.assertNotIn("shutil.rmtree(OPS_ROOT)", body)
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
        self.assertEqual(distribution["driver"], DRIVER)
        for key in (
            "preferred_real_kaggle_mount_paths_embedded", "preferred_paths_still_hash_structure_verified",
            "two_stage_torchvision_pretrained_provenance", "frozen_g1a_sealer_missing_global_compatibility",
            "per_account_singleton_guard", "duplicate_master_invocation_mirrors_primary_result",
            "process_isolated_runtime_checkout", "idempotent_git_evidence_publication",
            "final_full_bundle_roundtrip_required", "science_checkout_clean_rechecked_after_g1a_g2a_control",
        ):
            self.assertTrue(distribution[key])
        self.assertFalse(distribution["duplicate_master_invocation_runs_stages"])
        self.assertFalse(distribution["frozen_science_source_modified_for_compatibility"])
        self.assertFalse(distribution["scientific_result_steps_auto_retried"])
        self.assertEqual(freeze["runtime_candidate_sha"], RUNTIME_SHA)
        self.assertEqual(freeze["runtime_branch"], RUNTIME_BRANCH)
        self.assertEqual(freeze["qa_conclusion"], "success")
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
