from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SCIENCE_SHA = "9a72e9466a9a3e7429e0e36a028edac662f83146"
AUTHORITY = "EAAI-JE-SDL-v2.1-QA"
RELEASE_BRANCH = "ops-tracka-kaggle-launch-v1-20260913"
NOTEBOOKS = ["01_TRACKA_V12_G1A_SEAL.ipynb","02_TRACKA_V12_G2A_K1.ipynb","03_TRACKA_V12_G2A_K2.ipynb","04_TRACKA_V12_G2A_K3.ipynb","05_TRACKA_V12_SEAL_G2A_AND_SCIENCE_GO.ipynb","06_TRACKA_V12_SCIENCE_K1.ipynb","07_TRACKA_V12_SCIENCE_K2.ipynb","08_TRACKA_V12_SCIENCE_K3.ipynb"]

def text(path: str) -> str:
    return (OPS / path).read_text(encoding="utf-8")

class TrackAV12KaggleOperatorTests(unittest.TestCase):
    def test_operator_contract_matches_frozen_science(self):
        contract = json.loads(text("OPERATOR_CONTRACT.json"))
        self.assertEqual(contract["status"], "LOCKED_OPERATOR_CONTRACT")
        self.assertEqual(contract["scientific_source_sha"], SCIENCE_SHA)
        self.assertEqual(contract["authority_id"], AUTHORITY)
        self.assertEqual(contract["operator_schema_version"], "2.1")
        self.assertEqual(contract["operator_release_branch"], RELEASE_BRANCH)
        self.assertEqual(contract["notebooks"], NOTEBOOKS)
        self.assertFalse(contract["rules"]["scientific_source_may_follow_operator_head"])
        self.assertFalse(contract["rules"]["protected_test_open_before_track_a_closure"])
        self.assertFalse(contract["rules"]["external_prediction_open_before_track_a_closure"])
        self.assertTrue(contract["rules"]["g2a_requires_unique_private_kaggle_durability"])
        self.assertTrue(contract["rules"]["science_requires_final_durability_bound_go"])
        self.assertFalse(contract["rules"]["ddp_allowed"])
        self.assertFalse(contract["rules"]["dataparallel_allowed"])
        self.assertFalse(contract["rules"]["fsdp_allowed"])
    def test_all_notebooks_are_valid_frozen_and_release_bound(self):
        for name in NOTEBOOKS:
            nb = json.loads(text(name))
            self.assertEqual(nb["nbformat"], 4)
            self.assertEqual(nb["nbformat_minor"], 5)
            meta = nb["metadata"]["cropcop_operator"]
            self.assertEqual(meta["science_sha"], SCIENCE_SHA)
            self.assertEqual(meta["authority"], AUTHORITY)
            self.assertEqual(meta["operator_release_branch"], RELEASE_BRANCH)
            joined = json.dumps(nb)
            self.assertIn(RELEASE_BRANCH, joined)
            self.assertNotIn("GITHUB_TOKEN", joined)
            self.assertNotIn("CROPCOP_GITHUB_TOKEN", joined)
    def test_account_notebooks_have_exact_identity(self):
        mapping={"02_TRACKA_V12_G2A_K1.ipynb":"K1","03_TRACKA_V12_G2A_K2.ipynb":"K2","04_TRACKA_V12_G2A_K3.ipynb":"K3","06_TRACKA_V12_SCIENCE_K1.ipynb":"K1","07_TRACKA_V12_SCIENCE_K2.ipynb":"K2","08_TRACKA_V12_SCIENCE_K3.ipynb":"K3"}
        for name,account in mapping.items(): self.assertEqual(json.loads(text(name))["metadata"]["cropcop_operator"]["account_id"],account)
    def test_all_canonical_drivers_import_v2(self):
        for name in ("g1a_driver.py","g2a_account_driver.py","seal_go_driver.py","science_account_driver.py"):
            source=text(name); self.assertIn("from tracka_v12_kaggle_operator_v2 import",source); self.assertNotIn("from tracka_v12_kaggle_operator import (",source)
    def test_v2_fixes_json_serialization_and_g1a_schema(self):
        source=text("tracka_v12_kaggle_operator_v2.py")
        self.assertIn('OPERATOR_SCHEMA_VERSION = "2.1"',source)
        self.assertIn("payload = json.dumps(mapping, sort_keys=True)",source)
        self.assertNotIn("json.dumps(json.dumps(mapping))",source)
        self.assertIn('payload.get("source_git_sha") == SCIENCE_SHA',source)
    def test_helper_constants_match_contract(self):
        source=text("tracka_v12_kaggle_operator.py"); contract=json.loads(text("OPERATOR_CONTRACT.json"))
        for value in (SCIENCE_SHA,contract["dataset"]["manifest_sha256"],contract["dataset"]["class_map_sha256"],contract["principal_g1_seal_sha256"],contract["r13"]["sha256"],contract["r13"]["hf_commit"]): self.assertIn(value,source)
        self.assertIn(f'R13_BYTES = {contract["r13"]["bytes"]}',source)
    def test_g2a_profile_mapping_exact(self):
        contract=json.loads(text("OPERATOR_CONTRACT.json")); source=text("tracka_v12_kaggle_operator.py")
        profiles=[]
        for account,rows in contract["g2a_profiles"].items():
            self.assertIn(f'"{account}"',source)
            for cid,eid,slot,gpu in rows: self.assertIn(cid,source); self.assertIn(eid,source); self.assertIn(slot,source); profiles.append(cid)
        self.assertEqual(len(profiles),5); self.assertEqual(len(set(profiles)),5)
    def test_g2a_driver_is_non_scientific_and_durable(self):
        source=text("g2a_account_driver.py")
        for required in ('"--mode", "calibration"','"--durable-store-kind", "kaggle-dataset"','"--durable-required"','"--initial-steps", "24"','"--resume-steps", "8"','payload.get("validation_enabled") is not False','payload.get("scientific_metric_computed") is not False'): self.assertIn(required,source)
        self.assertIn("CUDA_VISIBLE_DEVICES",text("tracka_v12_kaggle_operator.py")); self.assertNotIn("--science-authorization",source)
    def test_go_and_science_use_canonical_go_schema(self):
        go=text("seal_go_driver.py"); science=text("science_account_driver.py")
        self.assertIn("seal_tracka_v12_science_go_v123.py",go); self.assertIn('go.get("status") != "GO"',go); self.assertNotIn('go.get("science_authorized")',go); self.assertIn("validate_science_authorization",go)
        self.assertIn("run_tracka_v12_account.py",science); self.assertIn("validate_science_authorization",science); self.assertIn('go.get("status") != "GO"',science); self.assertIn("validate_g2a_v122_barrier",science); self.assertIn("validate_scheduler_freeze_v122",science); self.assertNotIn("--publish-evidence",science); self.assertIn("wrong_owner",science); self.assertIn("continuation_is_technical",science)
    def test_g1a_driver_uses_frozen_seal_key(self):
        source=text("g1a_driver.py"); self.assertIn('seal.get("source_git_sha") != SCIENCE_SHA',source); self.assertNotIn('seal.get("source_git_commit") != SCIENCE_SHA',source)
    def test_protected_surfaces_not_requested(self):
        combined="\n".join(text(n) for n in ("g1a_driver.py","g2a_account_driver.py","seal_go_driver.py","science_account_driver.py"))
        self.assertNotIn("DS-V1-TEST-CONSUMED",combined); self.assertNotIn("--test",combined); self.assertNotIn("track_b",combined.lower()); self.assertNotIn("track_c",combined.lower())
if __name__ == "__main__": unittest.main()
