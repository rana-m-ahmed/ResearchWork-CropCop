from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tracka_v12_kaggle_operator_v8 as active
import master_control_v12 as control
import master_launch_guard_v12 as guard

SCIENCE_SHA = "56023042e57758591df9babb3438f191dbe10312"
ATTEST = OPS / "attestations_v10"


class TrackAV12KaggleMasterV12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code = json.loads((ATTEST / "tracka-v12-pre-science-code-attestation.json").read_text(encoding="utf-8"))
        cls.lock = json.loads((ATTEST / "tracka-v12-exact-head-lock-runtime-attestation.json").read_text(encoding="utf-8"))

    def test_science_identity_is_unchanged(self):
        self.assertEqual(active.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(active.SCIENCE_SHA_V8, SCIENCE_SHA)

    def test_exact_hardened_code_attestation_shape_is_recognized(self):
        self.assertEqual(self.code["schema_version"], "1.1")
        self.assertEqual(self.code["source_git_commit"], SCIENCE_SHA)
        self.assertEqual(self.code["status"], "PASS")
        static = self.code["static_pre_science_gates"]
        self.assertEqual(static[control.CODE_EXTRA_STATIC_GATE], "PASS")
        self.assertTrue(control._verify_embedded_hash(self.code, "attestation_sha256"))

    def test_exact_lock_runtime_attestation_uses_r13_prefixed_gate(self):
        self.assertEqual(self.lock["schema_version"], "1.1")
        self.assertEqual(self.lock["source_git_commit"], SCIENCE_SHA)
        self.assertEqual(self.lock["status"], "PASS")
        self.assertEqual(self.lock[control.LOCK_RUNTIME_QUALIFICATION_FIELD], "PASS")
        self.assertNotIn(control.AUTH_RUNTIME_QUALIFICATION_GATE, self.lock)
        self.assertTrue(control._verify_embedded_hash(self.lock, "attestation_sha256"))

    def test_compatibility_mapping_is_exact_and_narrow(self):
        self.assertEqual(control.CODE_EXTRA_STATIC_GATE, "teacher_factory_root_binding_contract")
        self.assertEqual(control.LOCK_RUNTIME_QUALIFICATION_FIELD, "r13_v121_runtime_qualification")
        self.assertEqual(control.AUTH_RUNTIME_QUALIFICATION_GATE, "v121_runtime_qualification")
        self.assertNotEqual(control.LOCK_RUNTIME_QUALIFICATION_FIELD, control.AUTH_RUNTIME_QUALIFICATION_GATE)

    def test_driver_only_rebinds_control_builder(self):
        text = (OPS / "master_account_driver_v12.py").read_text(encoding="utf-8")
        self.assertIn("import master_account_driver_v11 as base", text)
        self.assertIn("from master_control_v12 import build_control_k1", text)
        self.assertIn("base.build_control_k1 = build_control_k1", text)
        for forbidden in ("seed", "learning_rate", "weight_decay", "microbatch", "DS-V1-TEST"):
            self.assertNotIn(forbidden, text)

    def test_v12_guard_is_isolated_and_launches_v12_driver(self):
        text = (OPS / "master_launch_guard_v12.py").read_text(encoding="utf-8")
        self.assertIn("master_account_driver_v12.py", text)
        self.assertIn(".cropcop_tracka_master_guard_v12", text)
        self.assertNotIn("master_account_driver_v11.py", text)
        status = {
            "schema_version": guard.STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": "K1",
            "science_sha": SCIENCE_SHA,
            "operator_runtime_sha": "r" * 40,
            "return_code": 2,
        }
        self.assertEqual(guard._matching_terminal_code(status, account_id="K1", runtime_sha="r" * 40), 2)

    def test_control_bridge_preserves_protected_surface_closure(self):
        names = ("master_control_v12.py", "master_account_driver_v12.py", "master_launch_guard_v12.py")
        text = "\n".join((OPS / name).read_text(encoding="utf-8") for name in names)
        for forbidden in ("DS-V1-TEST-CONSUMED", "DS-EXT-*-SEALED", "external_predictions_open", "track_c_candidate_runtime_open"):
            self.assertNotIn(forbidden, text)
        self.assertIn('"protected_test_accessed": False', text)
        self.assertIn('"external_surface_accessed": False', text)

    def test_runtime_compatibility_report_explicitly_disclaims_science_change(self):
        text = (OPS / "master_control_v12.py").read_text(encoding="utf-8")
        self.assertIn("Runtime-only compatibility bridge", text)
        self.assertIn("No scientific model, seed, data, objective, selector, G1A, G2A", text)


if __name__ == "__main__":
    unittest.main()
