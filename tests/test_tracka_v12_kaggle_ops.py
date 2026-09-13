from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SCIENCE_SHA = "9a72e9466a9a3e7429e0e36a028edac662f83146"
RUNTIME_SHA = "481d347022c4dececdd81f2d676570a79f6c0105"
RUNTIME_BRANCH = "ops-tracka-kaggle-master-runtime-v3-fix4b-481d347"
AUTHORITY = "EAAI-JE-SDL-v2.1-QA"
NOTEBOOKS = {
    "TRACKA_V12_MASTER_K1.ipynb": "K1",
    "TRACKA_V12_MASTER_K2.ipynb": "K2",
    "TRACKA_V12_MASTER_K3.ipynb": "K3",
}


def text(path: str) -> str:
    return (OPS / path).read_text(encoding="utf-8")


class TrackAV12KaggleOperatorDistributionTests(unittest.TestCase):
    def test_operator_contract_is_v3_and_exact(self):
        contract = json.loads(text("OPERATOR_CONTRACT.json"))
        self.assertEqual(contract["schema_version"], "3.4")
        self.assertEqual(contract["status"], "LOCKED_MASTER_OPERATOR_CONTRACT")
        self.assertEqual(contract["scientific_source_sha"], SCIENCE_SHA)
        self.assertEqual(contract["operator_runtime_sha"], RUNTIME_SHA)
        self.assertEqual(contract["operator_runtime_branch"], RUNTIME_BRANCH)
        self.assertEqual(contract["authority_id"], AUTHORITY)
        self.assertEqual(contract["notebooks"], list(NOTEBOOKS))
        self.assertEqual(contract["kaggle"]["g2a_distinct_private_dataset_count"], 5)
        self.assertEqual(contract["kaggle"]["scientific_distinct_private_dataset_count"], 11)
        self.assertEqual(contract["kaggle"]["canonical_g1a_private_dataset_count"], 1)
        self.assertTrue(contract["dataset"]["fail_fast_before_dependency_install"])
        self.assertTrue(contract["dataset"]["diagnose_mounted_candidate_hashes"])
        self.assertTrue(contract["dataset"]["nested_split_root_discovery"])
        self.assertEqual(contract["dataset"]["root_qualification_surfaces"], ["train", "val"])
        self.assertFalse(contract["dataset"]["protected_test_opened_for_root_qualification"])
        self.assertTrue(contract["dataset"]["resolved_paths_bound_for_downstream_reuse"])
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

    def test_master_notebooks_are_clean_pinned_and_account_specific(self):
        for name, account in NOTEBOOKS.items():
            with self.subTest(name=name):
                nb = json.loads(text(name))
                self.assertEqual(nb["nbformat"], 4)
                self.assertEqual(nb["nbformat_minor"], 5)
                meta = nb["metadata"]["cropcop_operator"]
                self.assertEqual(meta["schema_version"], "3.4")
                self.assertEqual(meta["account_id"], account)
                self.assertEqual(meta["science_sha"], SCIENCE_SHA)
                self.assertEqual(meta["operator_runtime_sha"], RUNTIME_SHA)
                self.assertEqual(meta["operator_runtime_branch"], RUNTIME_BRANCH)
                self.assertEqual(meta["authority"], AUTHORITY)
                joined = json.dumps(nb)
                self.assertIn(RUNTIME_SHA, joined)
                self.assertIn(RUNTIME_BRANCH, joined)
                self.assertIn("master_account_driver_v4.py", joined)
                self.assertIn(f"str(driver), '{account}'", joined)
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
        self.assertNotIn("CROPCOP_PRINCIPAL_G1", text("TRACKA_V12_MASTER_K2.ipynb"))
        self.assertNotIn("CROPCOP_PRINCIPAL_G1", text("TRACKA_V12_MASTER_K3.ipynb"))

    def test_distribution_documents_same_runtime(self):
        distribution = json.loads(text("MASTER_NOTEBOOK_DISTRIBUTION.json"))
        freeze = json.loads(text("MASTER_RUNTIME_FREEZE.json"))
        marker = text(".runtime-freeze-marker")
        self.assertEqual(distribution["operator_runtime_sha"], RUNTIME_SHA)
        self.assertEqual(distribution["operator_runtime_branch"], RUNTIME_BRANCH)
        self.assertTrue(distribution["partial_bundle_incremental_publication"])
        self.assertTrue(distribution["final_full_bundle_roundtrip_required"])
        self.assertEqual(freeze["runtime_candidate_sha"], RUNTIME_SHA)
        self.assertEqual(freeze["runtime_branch"], RUNTIME_BRANCH)
        self.assertEqual(freeze["qa_test_count"], 53)
        self.assertIn(RUNTIME_SHA, marker)
        self.assertIn(RUNTIME_BRANCH, marker)

    def test_protected_surfaces_remain_closed_in_master_runtime(self):
        combined = "\n".join(text(name) for name in (
            "master_account_driver_v4.py", "master_g1a.py", "master_g2a.py", "master_control.py", "master_science_v4.py"
        ))
        self.assertNotIn("DS-V1-TEST-CONSUMED", combined)
        self.assertNotIn("--test", combined)
        self.assertNotIn("track_b", combined.lower())
        self.assertNotIn("track_c", combined.lower())


if __name__ == "__main__":
    unittest.main()
