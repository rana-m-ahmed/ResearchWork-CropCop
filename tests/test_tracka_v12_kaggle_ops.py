from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SCIENCE_SHA = "9a72e9466a9a3e7429e0e36a028edac662f83146"
RUNTIME_SHA = "5654f35fa52c9b6ae6c28f062969c9ebd65af3fa"
RUNTIME_BRANCH = "ops-tracka-kaggle-master-runtime-v3-fix1-5654f35"
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
        self.assertEqual(contract["schema_version"], "3.1")
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
                self.assertEqual(meta["schema_version"], "3.1")
                self.assertEqual(meta["account_id"], account)
                self.assertEqual(meta["science_sha"], SCIENCE_SHA)
                self.assertEqual(meta["operator_runtime_sha"], RUNTIME_SHA)
                self.assertEqual(meta["operator_runtime_branch"], RUNTIME_BRANCH)
                self.assertEqual(meta["authority"], AUTHORITY)
                joined = json.dumps(nb)
                self.assertIn(RUNTIME_SHA, joined)
                self.assertIn(RUNTIME_BRANCH, joined)
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
        readme = text("README.md")
        master_readme = text("README_MASTER_V3.md")
        contract = json.loads(text("MASTER_OPERATOR_CONTRACT_V3.json"))
        for body in (readme, master_readme, json.dumps(contract)):
            self.assertIn(SCIENCE_SHA, body)
        self.assertIn(RUNTIME_SHA, readme)
        self.assertIn(RUNTIME_BRANCH, readme)
        self.assertEqual(contract["scientific_source_sha"], SCIENCE_SHA)
        self.assertEqual(contract["operator_model"], "three_account_master_notebooks")

    def test_protected_surfaces_remain_closed_in_master_runtime(self):
        combined = "\n".join(text(name) for name in (
            "master_account_driver.py", "master_g1a.py", "master_g2a.py", "master_control.py", "master_science.py"
        ))
        self.assertNotIn("DS-V1-TEST-CONSUMED", combined)
        self.assertNotIn("--test", combined)
        self.assertNotIn("track_b", combined.lower())
        self.assertNotIn("track_c", combined.lower())


if __name__ == "__main__":
    unittest.main()
