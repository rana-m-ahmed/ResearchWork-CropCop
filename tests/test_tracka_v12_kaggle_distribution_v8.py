from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SCIENCE = "05ac7084a6be2fecd9c370477340ee0c8c4769bc"
RUNTIME = "2f127cbb61d752eec22c3bfd3ecd527a3a81c283"
RUNTIME_BRANCH = "ops-tracka-kaggle-master-runtime-v8-2f127cb"
NOTEBOOKS = [
    "TRACKA_V12_MASTER_K1.ipynb",
    "TRACKA_V12_MASTER_K2.ipynb",
    "TRACKA_V12_MASTER_K3.ipynb",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrackAV12DistributionV8Tests(unittest.TestCase):
    def setUp(self):
        self.distribution = json.loads((OPS / "MASTER_NOTEBOOK_DISTRIBUTION.json").read_text(encoding="utf-8"))
        self.freeze = json.loads((OPS / "MASTER_RUNTIME_FREEZE.json").read_text(encoding="utf-8"))
        self.contract = json.loads((OPS / "OPERATOR_CONTRACT.json").read_text(encoding="utf-8"))

    def test_exactly_three_master_notebooks_are_canonical(self):
        actual = sorted(path.name for path in OPS.glob("*.ipynb"))
        self.assertEqual(actual, NOTEBOOKS)
        self.assertEqual(self.distribution["normal_operator_notebook_count"], 3)
        self.assertEqual(self.distribution["normal_operator_notebooks"], NOTEBOOKS)

    def test_every_notebook_is_exactly_runtime_and_science_pinned(self):
        for account_id, name in zip(("K1", "K2", "K3"), NOTEBOOKS, strict=True):
            path = OPS / name
            nb = json.loads(path.read_text(encoding="utf-8"))
            meta = nb["metadata"]["cropcop_operator"]
            self.assertEqual(meta["schema_version"], "4.0")
            self.assertEqual(meta["account_id"], account_id)
            self.assertEqual(meta["science_sha"], SCIENCE)
            self.assertEqual(meta["operator_runtime_sha"], RUNTIME)
            self.assertEqual(meta["operator_runtime_branch"], RUNTIME_BRANCH)
            self.assertEqual(meta["launcher"], "master_launch_guard_v8.py")
            self.assertEqual(meta["driver"], "master_account_driver_v8.py")
            self.assertTrue(meta["expected_account_binding_required"])
            source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
            self.assertIn(RUNTIME, source)
            self.assertIn(RUNTIME_BRANCH, source)
            self.assertIn("master_launch_guard_v8.py", source)
            self.assertIn("CROPCOP_EXPECTED_KAGGLE_USERNAME", source)
            self.assertNotIn("master_account_driver_v5.py", source)
            self.assertNotIn("master_account_driver_v6.py", source)
            self.assertNotIn("master_account_driver_v7.py", source)
            self.assertNotIn("master_g1a_sealer_compat_v6", source)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile("".join(cell.get("source") or []), f"{name}:cell", "exec")

    def test_notebook_hash_manifest_matches_exact_bytes(self):
        expected = self.distribution["notebook_sha256"]
        self.assertEqual(set(expected), set(NOTEBOOKS))
        for name in NOTEBOOKS:
            self.assertEqual(expected[name], sha256(OPS / name))

    def test_k1_only_notebook_mentions_principal_g1(self):
        sources = {}
        for name in NOTEBOOKS:
            nb = json.loads((OPS / name).read_text(encoding="utf-8"))
            sources[name] = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
        self.assertIn("CROPCOP_PRINCIPAL_G1", sources["TRACKA_V12_MASTER_K1.ipynb"])
        self.assertNotIn("CROPCOP_PRINCIPAL_G1", sources["TRACKA_V12_MASTER_K2.ipynb"])
        self.assertNotIn("CROPCOP_PRINCIPAL_G1", sources["TRACKA_V12_MASTER_K3.ipynb"])

    def test_distribution_freeze_and_contract_are_one_identity(self):
        for payload, source_key, runtime_key, branch_key in (
            (self.distribution, "science_source_sha", "operator_runtime_sha", "operator_runtime_branch"),
            (self.freeze, "science_source_sha", "runtime_candidate_sha", "runtime_branch"),
            (self.contract, "scientific_source_sha", "operator_runtime_sha", "operator_runtime_branch"),
        ):
            self.assertEqual(payload[source_key], SCIENCE)
            self.assertEqual(payload[runtime_key], RUNTIME)
            self.assertEqual(payload[branch_key], RUNTIME_BRANCH)
        self.assertEqual(self.freeze["status"], "PASS")
        self.assertEqual(self.freeze["qa_conclusion"], "success")
        self.assertEqual(self.freeze["qa_workflow_run_id"], 34936176259)
        self.assertEqual(self.contract["status"], "LOCKED_MASTER_OPERATOR_CONTRACT")

    def test_release_integrity_and_durability_controls_are_frozen(self):
        self.assertTrue(self.distribution["release_integrity_before_expensive_work"])
        self.assertTrue(self.distribution["generation_aware_kaggle_durability"])
        self.assertTrue(self.distribution["session_aware_dependency_deadline"])
        self.assertTrue(self.distribution["g1a_failed_handoff_fail_fast"])
        self.assertTrue(self.distribution["g2a_account_failed_status_fail_fast"])
        self.assertEqual(self.distribution["exact_head_code_attestation_sha256"], "cd8a23ca6a5ebdba18e8466d8578d2646723c349b9428b8d8890aac7c62ab187")
        self.assertEqual(self.distribution["exact_head_lock_runtime_attestation_sha256"], "046814249b119f6d520df58095bf951b13e35fcc09f995599c36202f8b0bed7f")
        self.assertTrue(self.contract["durability"]["kaggle_generation_must_advance_before_sync_success"])
        self.assertTrue(self.contract["durability"]["generation_marker_roundtrip_required"])
        self.assertTrue(self.contract["durability"]["checkpoint_index_hash_bound"])

    def test_protected_surfaces_and_parallelism_remain_closed(self):
        self.assertFalse(self.distribution["protected_test_open_before_track_a_closure"])
        self.assertFalse(self.distribution["external_prediction_open_before_track_a_closure"])
        rules = self.contract["rules"]
        self.assertFalse(rules["protected_test_open_before_track_a_closure"])
        self.assertFalse(rules["external_prediction_open_before_track_a_closure"])
        self.assertFalse(rules["ddp_allowed"])
        self.assertFalse(rules["dataparallel_allowed"])
        self.assertFalse(rules["fsdp_allowed"])

    def test_runtime_marker_and_note_match_freeze(self):
        marker = (OPS / ".runtime-freeze-marker").read_text(encoding="utf-8")
        self.assertIn(f"runtime_sha={RUNTIME}", marker)
        self.assertIn(f"runtime_branch={RUNTIME_BRANCH}", marker)
        self.assertIn(f"science_sha={SCIENCE}", marker)
        note = (OPS / "RUNTIME_BRANCH_NOTE.md").read_text(encoding="utf-8")
        self.assertIn(RUNTIME, note)
        self.assertIn(RUNTIME_BRANCH, note)
        self.assertIn(SCIENCE, note)


if __name__ == "__main__":
    unittest.main()
