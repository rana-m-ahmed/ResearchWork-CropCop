from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tracka_v12_kaggle_operator as v1
import tracka_v12_kaggle_operator_v2 as v2
import tracka_v12_kaggle_operator_v3 as v3
import tracka_v12_kaggle_operator_v8 as v8
import master_account_driver_v8 as driver
import master_g1a_v8 as g1a
import master_g2a_v8 as g2a
import master_launch_guard_v8 as guard
import master_wait_v8 as wait8
from master_attestations_v8 import verified_attestation_paths

SCIENCE_SHA = "05ac7084a6be2fecd9c370477340ee0c8c4769bc"


class TrackAV12KaggleMasterV8Tests(unittest.TestCase):
    def test_v8_rebinds_all_inherited_science_authorities(self):
        self.assertEqual(v8.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v1.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v2.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v3.SCIENCE_SHA, SCIENCE_SHA)

    def test_packaged_exact_head_attestations_are_exact_and_current(self):
        code, lock = verified_attestation_paths()
        self.assertEqual(v8.sha256_file(code), "cd8a23ca6a5ebdba18e8466d8578d2646723c349b9428b8d8890aac7c62ab187")
        self.assertEqual(v8.sha256_file(lock), "046814249b119f6d520df58095bf951b13e35fcc09f995599c36202f8b0bed7f")
        cp = json.loads(code.read_text(encoding="utf-8")); lp = json.loads(lock.read_text(encoding="utf-8"))
        for payload in (cp, lp):
            self.assertEqual(payload["source_git_commit"], SCIENCE_SHA)
            self.assertEqual(payload["pull_request_head_sha"], SCIENCE_SHA)
            self.assertEqual(payload["status"], "PASS")
        self.assertEqual(cp["static_pre_science_gates"]["kaggle_generation_durability_contract"], "PASS")
        self.assertEqual(cp["static_pre_science_gates"]["g1a_runtime_global_resolution_contract"], "PASS")
        self.assertIn("journal_extension/src/cropcop_je/persistence_v8.py", cp["implementation_sha256"])
        self.assertFalse(lp["science_authorized"])

    def test_release_gate_precedes_input_resolution_and_expensive_work(self):
        text = (OPS / "master_account_driver_v8.py").read_text(encoding="utf-8")
        release = text.index('stage("RELEASE_INTEGRITY_PREFLIGHT"')
        verify = text.index("verify_release_integrity()")
        inputs = text.index('stage("FROZEN_INPUT_RESOLUTION"')
        checkout = text.index('stage("SCIENCE_SOURCE_AND_GITHUB_PREFLIGHT"')
        stack = text.index('stage("EXACT_EXECUTION_STACK"')
        g1 = text.index('stage("CANONICAL_G1A"')
        self.assertLess(release, inputs)
        self.assertLess(verify, inputs)
        self.assertLess(inputs, checkout)
        self.assertLess(checkout, stack)
        self.assertLess(stack, g1)

    def test_v8_direct_g1a_and_g2a_paths_have_no_compatibility_injection(self):
        combined = "\n".join((OPS / name).read_text(encoding="utf-8") for name in (
            "master_account_driver_v8.py", "master_g1a_v8.py", "master_g2a_v8.py",
            "master_control_v8.py", "master_launch_guard_v8.py",
        ))
        self.assertIn("seal_tracka_v12_g1a.py", combined)
        for forbidden in ("master_g1a_sealer_compat_v6", "init_globals", "runpy.run_path", "TORCHVISION_VERSION ="):
            self.assertNotIn(forbidden, combined)

    def test_wait_budget_uses_global_session_deadline_not_thirty_minutes(self):
        with mock.patch.dict(os.environ, {
            "CROPCOP_NOTEBOOK_STARTED_MONOTONIC": "1000",
            "CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS": "43200",
            "CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS": "3600",
        }, clear=False), mock.patch.object(wait8.time, "monotonic", return_value=2000):
            self.assertEqual(wait8.dependency_deadline_monotonic(), 40600.0)
            self.assertEqual(wait8.remaining_dependency_seconds(), 38600.0)
            self.assertFalse(wait8.dependency_wait_expired())

    def test_g1a_failed_handoff_is_non_authorizing(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g1a, "operator_runtime_head", return_value="r" * 40):
            path = Path(td) / "handoff.json"
            payload = g1a.write_handoff(path, status="FAILED", locator="owner/dataset", failure_code=g1a.G1A_FAILURE_CODE)
            self.assertEqual(payload["status"], "FAILED")
            self.assertIsNone(payload["g1a_seal_sha256"])
            self.assertFalse(payload["science_authorized"])
            self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)

    def test_g2a_failed_status_is_non_authorizing_and_bound(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g2a, "operator_runtime_head", return_value="r" * 40):
            path = Path(td) / "status.json"
            payload = g2a.write_account_status(
                path, account_id="K2", status="FAILED", g1a_seal_sha256="a" * 64,
                failure_code=g2a.G2A_FAILURE_CODE,
            )
            self.assertEqual(payload["status"], "FAILED")
            self.assertFalse(payload["science_authorized"])
            self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
            self.assertEqual(payload["g1a_seal_sha256"], "a" * 64)

    def test_expected_account_binding_is_mandatory(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(v8.OperatorError):
                driver.assert_expected_account("K2", "some-user")
        with mock.patch.dict(os.environ, {"CROPCOP_EXPECTED_KAGGLE_USERNAME": "correct-user"}, clear=True):
            with self.assertRaises(v8.OperatorError):
                driver.assert_expected_account("K2", "wrong-user")
            driver.assert_expected_account("K2", "correct-user")

    def test_guard_terminal_reuse_requires_exact_science_runtime_account(self):
        status = {
            "schema_version": guard.STATUS_SCHEMA, "state": "FINISHED", "account_id": "K1",
            "science_sha": SCIENCE_SHA, "operator_runtime_sha": "r" * 40, "return_code": 0,
        }
        self.assertEqual(guard._matching_terminal_code(status, account_id="K1", runtime_sha="r" * 40), 0)
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K2", runtime_sha="r" * 40))
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K1", runtime_sha="x" * 40))

    def test_protected_surfaces_remain_closed(self):
        combined = "\n".join((OPS / name).read_text(encoding="utf-8") for name in (
            "master_account_driver_v8.py", "master_g1a_v8.py", "master_g2a_v8.py", "master_control_v8.py"
        ))
        self.assertNotIn("DS-V1-TEST-CONSUMED", combined)
        self.assertNotIn("--test", combined)
        self.assertNotIn("track_b", combined.lower())
        self.assertNotIn("track_c", combined.lower())


if __name__ == "__main__":
    unittest.main()
