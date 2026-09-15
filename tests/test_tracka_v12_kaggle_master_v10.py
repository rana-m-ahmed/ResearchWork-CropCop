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
import tracka_v12_kaggle_operator_v8 as active
import master_attestations_v8 as attest
import master_continuation_v10 as cont
import master_account_driver_v10 as driver
import master_launch_guard_v10 as guard
import master_g1a_v8 as g1a
import master_g2a_v8 as g2a

SCIENCE_SHA = "56023042e57758591df9babb3438f191dbe10312"
MANIFEST_SHA256 = "dedf8a3b9ad29222b6bbbee669442012f0171ec14d89dca5f743ac0d7ef4ffd3"
CODE_MEMBER_SHA256 = "d07244284d0ca77b8f96412c11ef2a5d58ca960657ddff089bba61c7a6e43e63"
LOCK_MEMBER_SHA256 = "38b24403c98bdb55649d76e4ab58b362ec307e6665c1bbcb936d58a5bb38b863"
TEACHER_SHA256 = "4ec41477264387cde03f7fab8f7df4b9480223b32f8554af5c6198687a568f02"


class TrackAV12KaggleMasterV10Tests(unittest.TestCase):
    def test_exact_science_authority_and_schema(self):
        self.assertEqual(active.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v1.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v2.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v3.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(active.OPERATOR_SCHEMA_VERSION, "4.6")

    def test_release_manifest_and_packaged_attestations_are_exact(self):
        path, payload = attest.verified_release_attestation_manifest()
        self.assertEqual(active.sha256_file(path), MANIFEST_SHA256)
        self.assertEqual(payload["schema_version"], "1.2")
        self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
        self.assertFalse(payload["science_authorized"])
        self.assertFalse(payload["protected_test_accessed"])
        self.assertFalse(payload["external_surface_accessed"])
        self.assertEqual(payload["remaining_scientific_states"], 11)
        self.assertEqual(payload["code_attestation"]["member_sha256"], CODE_MEMBER_SHA256)
        self.assertEqual(payload["lock_runtime_attestation"]["member_sha256"], LOCK_MEMBER_SHA256)
        teacher = payload["teacher_factory_root_contract"]
        self.assertEqual(teacher["gate"], "PASS")
        self.assertEqual(teacher["entrypoint"], "historical_dino_tiny:build_teacher")
        self.assertEqual(teacher["source_sha256"], TEACHER_SHA256)
        self.assertIn("teacher_factory_root_binding_contract", payload["required_static_pre_science_gates"])

        code, lock = attest.verified_attestation_paths()
        self.assertEqual(active.sha256_file(code), CODE_MEMBER_SHA256)
        self.assertEqual(active.sha256_file(lock), LOCK_MEMBER_SHA256)
        code_payload = active.load_json(code)
        self.assertEqual(code_payload["source_git_commit"], SCIENCE_SHA)
        self.assertEqual(code_payload["pull_request_head_sha"], SCIENCE_SHA)
        self.assertEqual(
            code_payload["static_pre_science_gates"]["teacher_factory_root_binding_contract"],
            "PASS",
        )
        self.assertEqual(
            code_payload["implementation_sha256"]["journal_extension/teacher_factory/historical_dino_tiny.py"],
            TEACHER_SHA256,
        )

    def test_attestations_materialize_without_actions_api_or_token(self):
        text = (OPS / "master_attestations_v8.py").read_text(encoding="utf-8")
        self.assertIn("attestations_v10", text)
        for forbidden in (
            "urllib.request",
            "urllib.error",
            "/actions/artifacts/",
            "api.github.com",
            "CROPCOP_GITHUB_TOKEN",
        ):
            self.assertNotIn(forbidden, text)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {}, clear=True):
            code, lock = attest.materialize_verified_attestations(td)
            self.assertEqual(active.sha256_file(code), CODE_MEMBER_SHA256)
            self.assertEqual(active.sha256_file(lock), LOCK_MEMBER_SHA256)

    def test_versioned_source_contains_teacher_factory_root_fix(self):
        self.assertEqual(active.SCIENCE_SHA, SCIENCE_SHA)
        manifest = active.load_json(attest.ATTESTATION_DIR / "RELEASE_ATTESTATION_MANIFEST.json")
        self.assertEqual(manifest["teacher_factory_root_contract"]["source_sha256"], TEACHER_SHA256)

    def test_try_collect_all_g2a_returns_none_without_polling_when_peer_missing(self):
        seal = {"g1a_seal_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(cont, "fetch_existing_g2a", return_value=None), \
             mock.patch.object(cont, "_fetch_current_account_status", return_value={"status": "PREPARING"}):
            result = cont.try_collect_all_g2a(Path(td), g1a_seal=seal, destination=Path(td) / "all")
        self.assertIsNone(result)

    def test_try_collect_all_g2a_fails_closed_on_current_peer_failure(self):
        seal = {"g1a_seal_sha256": "a" * 64}
        failed = {"status": "FAILED", "failure_code": g2a.G2A_FAILURE_CODE}
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(cont, "fetch_existing_g2a", return_value=None), \
             mock.patch.object(cont, "_fetch_current_account_status", return_value=failed):
            with self.assertRaises(active.OperatorError):
                cont.try_collect_all_g2a(Path(td), g1a_seal=seal, destination=Path(td) / "all")

    def test_try_acquire_worker_control_returns_none_without_polling(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(cont, "fetch_public_bundle", return_value=None):
            result = cont.try_acquire_control_worker(
                Path(td),
                g1a_seal={"g1a_seal_sha256": "a" * 64},
                master_root=Path(td) / "master",
            )
        self.assertIsNone(result)

    def test_driver_control_plane_uses_nonblocking_v10_helpers(self):
        text = (OPS / "master_account_driver_v10.py").read_text(encoding="utf-8")
        self.assertIn("try_collect_all_g2a", text)
        self.assertIn("try_acquire_control_worker", text)
        self.assertIn("controlled_dependency_continuation", text)
        self.assertNotIn("from master_g2a_v8 import collect_all_g2a", text)
        self.assertNotIn("summaries = collect_all_g2a(", text)
        self.assertNotIn("control_dir, control = acquire_control_worker(", text)
        self.assertIn("return 2", text)

    def test_controlled_continuation_is_terminal_for_current_batch_only(self):
        self.assertEqual(driver.controlled_dependency_continuation("K2", "control not ready"), 2)

    def test_v10_guard_invokes_v10_driver_and_binds_exact_identity(self):
        text = (OPS / "master_launch_guard_v10.py").read_text(encoding="utf-8")
        self.assertIn("master_account_driver_v10.py", text)
        self.assertIn(".cropcop_tracka_master_guard_v10", text)
        status = {
            "schema_version": guard.STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": "K2",
            "science_sha": SCIENCE_SHA,
            "operator_runtime_sha": "r" * 40,
            "return_code": 2,
        }
        self.assertEqual(guard._matching_terminal_code(status, account_id="K2", runtime_sha="r" * 40), 2)
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K1", runtime_sha="r" * 40))

    def test_expected_account_binding_remains_mandatory(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(active.OperatorError):
                driver.assert_expected_account("K2", "some-user")
        with mock.patch.dict(os.environ, {"CROPCOP_EXPECTED_KAGGLE_USERNAME": "correct-user"}, clear=True):
            with self.assertRaises(active.OperatorError):
                driver.assert_expected_account("K2", "wrong-user")
            driver.assert_expected_account("K2", "correct-user")

    def test_failed_g1a_and_g2a_markers_remain_non_authorizing(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g1a, "operator_runtime_head", return_value="r" * 40):
            p = g1a.write_handoff(Path(td) / "g1a.json", status="FAILED", locator="owner/dataset", failure_code=g1a.G1A_FAILURE_CODE)
            self.assertFalse(p["science_authorized"])
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g2a, "operator_runtime_head", return_value="r" * 40):
            p = g2a.write_account_status(Path(td) / "g2a.json", account_id="K2", status="FAILED", g1a_seal_sha256="a" * 64, failure_code=g2a.G2A_FAILURE_CODE)
            self.assertFalse(p["science_authorized"])

    def test_protected_surfaces_and_science_are_not_opened_by_runtime_fix(self):
        names = (
            "tracka_v12_kaggle_operator_v8.py",
            "master_attestations_v8.py",
            "master_continuation_v10.py",
            "master_account_driver_v10.py",
            "master_launch_guard_v10.py",
        )
        text = "\n".join((OPS / name).read_text(encoding="utf-8") for name in names)
        for forbidden in ("DS-V1-TEST-CONSUMED", "DS-EXT-*-SEALED", "master_g1a_sealer_compat_v6"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
