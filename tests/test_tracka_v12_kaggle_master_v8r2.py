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
import master_attestations_v8 as attest
import master_g1a_v8 as g1a
import master_g2a_v8 as g2a
import master_launch_guard_v8 as guard
import master_wait_v8 as wait8

SCIENCE_SHA = "4ced2fd7c764c07fa47fb57fbea38376d2ce61a4"
MANIFEST_SHA256 = "a89ad7368eaf6f685087cffe4e428dfb5295d0909dc60d448b871946f3e2b61f"
CODE_MEMBER_SHA256 = "5bfb757219996aa7df48e21fd0531124c60a86f6a0c8ad17840a2a3eb88cda2e"
LOCK_MEMBER_SHA256 = "49efe43b76423c82c6849d43d3f1bea2ff24503f65564e722d205cdac2a21fea"


class TrackAV12KaggleMasterV8R2Tests(unittest.TestCase):
    def test_exact_science_authority_rebinds_all_inherited_modules(self):
        self.assertEqual(v8.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v1.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v2.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v3.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v8.OPERATOR_SCHEMA_VERSION, "4.2")
        self.assertEqual(v8.R13_PARITY_CONTRACT_ID_V8, "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1")
        self.assertEqual(v8.R13_PARITY_REQUIRED_MAX_ABS_V8, 5e-5)

    def test_inherited_g1a_helpers_resolve_versioned_validator(self):
        self.assertIs(v1.validate_g1a_bundle_with_science, v8.validate_g1a_bundle_with_science)
        self.assertIs(v2.validate_g1a_bundle_with_science, v8.validate_g1a_bundle_with_science)
        self.assertIs(v3.validate_g1a_bundle_with_science, v8.validate_g1a_bundle_with_science)

    def test_release_manifest_is_exact_and_non_authorizing(self):
        path, payload = attest.verified_release_attestation_manifest()
        self.assertEqual(v8.sha256_file(path), MANIFEST_SHA256)
        self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
        self.assertEqual(payload["code_attestation"]["member_sha256"], CODE_MEMBER_SHA256)
        self.assertEqual(payload["lock_runtime_attestation"]["member_sha256"], LOCK_MEMBER_SHA256)
        self.assertEqual(payload["remaining_scientific_states"], 11)
        self.assertFalse(payload["science_authorized"])
        self.assertFalse(payload["protected_test_accessed"])
        self.assertFalse(payload["external_surface_accessed"])
        self.assertEqual(payload["r13_parity_contract"]["required_max_abs_difference"], 5e-5)
        self.assertEqual(payload["r13_parity_contract"]["historical_v1_2_required_max_abs_difference"], 1e-5)

    def test_active_g1a_route_executes_only_versioned_sealer(self):
        text = (OPS / "master_g1a_v8.py").read_text(encoding="utf-8")
        self.assertIn("seal_tracka_v12_g1a_v121.py", text)
        self.assertNotIn('str(repo / "journal_extension/scripts/seal_tracka_v12_g1a.py")', text)
        self.assertIn("validate_g1a_bundle_with_science", text)

    def test_active_science_route_executes_versioned_account_parent(self):
        text = (OPS / "master_science_v8.py").read_text(encoding="utf-8")
        self.assertIn("run_tracka_v12_account_v121.py", text)
        self.assertNotIn('str(repo / "journal_extension/kaggle/run_tracka_v12_account.py")', text)

    def test_active_control_route_materializes_exact_attestations_and_uses_v124_go(self):
        text = (OPS / "master_control_v8.py").read_text(encoding="utf-8")
        self.assertIn("materialize_verified_attestations", text)
        self.assertIn("seal_tracka_v12_science_go_v124.py", text)
        self.assertNotIn("verified_attestation_paths", text)
        self.assertNotIn("seal_tracka_v12_science_go_v123.py", text)

    def test_exact_attestation_materialization_precedes_stack_and_g1a(self):
        text = (OPS / "master_account_driver_v8.py").read_text(encoding="utf-8")
        source_stage = text.index('stage("SCIENCE_SOURCE_GITHUB_AND_ATTESTATION_PREFLIGHT"')
        materialize = text.index("materialize_verified_attestations")
        stack_stage = text.index('stage("EXACT_EXECUTION_STACK"')
        g1a_stage = text.index('stage("CANONICAL_G1A"')
        self.assertLess(source_stage, materialize)
        self.assertLess(materialize, stack_stage)
        self.assertLess(stack_stage, g1a_stage)

    def test_active_runtime_has_no_legacy_compatibility_injection(self):
        names = (
            "tracka_v12_kaggle_operator_v8.py",
            "master_attestations_v8.py",
            "master_g1a_v8.py",
            "master_g2a_v8.py",
            "master_control_v8.py",
            "master_science_v8.py",
            "master_account_driver_v8.py",
            "master_launch_guard_v8.py",
        )
        combined = "\n".join((OPS / name).read_text(encoding="utf-8") for name in names)
        for forbidden in (
            "master_g1a_sealer_compat_v6",
            "init_globals",
            "runpy.run_path",
            "DS-V1-TEST-CONSUMED",
            "--test",
        ):
            self.assertNotIn(forbidden, combined)

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
            "schema_version": guard.STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": "K1",
            "science_sha": SCIENCE_SHA,
            "operator_runtime_sha": "r" * 40,
            "return_code": 0,
        }
        self.assertEqual(guard._matching_terminal_code(status, account_id="K1", runtime_sha="r" * 40), 0)
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K2", runtime_sha="r" * 40))
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K1", runtime_sha="x" * 40))

    def test_wait_budget_uses_global_session_deadline(self):
        with mock.patch.dict(os.environ, {
            "CROPCOP_NOTEBOOK_STARTED_MONOTONIC": "1000",
            "CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS": "43200",
            "CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS": "3600",
        }, clear=False), mock.patch.object(wait8.time, "monotonic", return_value=2000):
            self.assertEqual(wait8.dependency_deadline_monotonic(), 40600.0)
            self.assertEqual(wait8.remaining_dependency_seconds(), 38600.0)
            self.assertFalse(wait8.dependency_wait_expired())

    def test_g1a_failed_handoff_is_non_authorizing_and_exact_source_bound(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g1a, "operator_runtime_head", return_value="r" * 40):
            path = Path(td) / "handoff.json"
            payload = g1a.write_handoff(path, status="FAILED", locator="owner/dataset", failure_code=g1a.G1A_FAILURE_CODE)
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
            self.assertIsNone(payload["g1a_seal_sha256"])
            self.assertFalse(payload["science_authorized"])

    def test_g2a_failed_status_is_non_authorizing_and_exact_source_bound(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g2a, "operator_runtime_head", return_value="r" * 40):
            path = Path(td) / "status.json"
            payload = g2a.write_account_status(
                path,
                account_id="K2",
                status="FAILED",
                g1a_seal_sha256="a" * 64,
                failure_code=g2a.G2A_FAILURE_CODE,
            )
            self.assertEqual(payload["status"], "FAILED")
            self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
            self.assertFalse(payload["science_authorized"])

    def test_protected_surfaces_remain_closed_in_active_runtime(self):
        combined = "\n".join((OPS / name).read_text(encoding="utf-8") for name in (
            "master_account_driver_v8.py",
            "master_g1a_v8.py",
            "master_g2a_v8.py",
            "master_control_v8.py",
            "master_science_v8.py",
        ))
        self.assertNotIn("DS-V1-TEST-CONSUMED", combined)
        self.assertNotIn("track_b", combined.lower())
        self.assertNotIn("track_c", combined.lower())


if __name__ == "__main__":
    unittest.main()
