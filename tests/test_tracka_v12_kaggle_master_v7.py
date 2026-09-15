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
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import master_g1a_v7 as g1a  # noqa: E402
import master_g2a_v7 as g2a  # noqa: E402
import master_launch_guard_v7 as guard  # noqa: E402
from master_attestations import verified_attestation_paths  # noqa: E402
from tracka_v12_kaggle_operator_v3 import SCIENCE_SHA, OperatorError  # noqa: E402


class TrackAV12KaggleMasterV7Tests(unittest.TestCase):
    def test_repaired_science_head_is_bound(self):
        self.assertEqual(SCIENCE_SHA, "e21a505792ddd91df55712e248391c79c55cf235")

    def test_driver_uses_direct_v7_g1a_and_g2a_paths(self):
        text = (OPS / "master_account_driver_v7.py").read_text(encoding="utf-8")
        self.assertIn("from master_g1a_v7 import", text)
        self.assertIn("from master_g2a_v7 import", text)
        self.assertNotIn("master_g1a_v6", text)
        self.assertNotIn("master_g1a_sealer_compat_v6", text)
        self.assertNotIn("init_globals", text)
        self.assertNotIn("runpy", text)

    def test_packaged_exact_head_attestations_match_frozen_bytes(self):
        code, lock = verified_attestation_paths()
        self.assertTrue(code.is_file())
        self.assertTrue(lock.is_file())

    def test_g1a_failed_handoff_is_non_authorizing(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g1a, "operator_runtime_head", return_value="c" * 40):
            path = Path(td) / g1a.G1A_HANDOFF_FILE
            payload = g1a.write_handoff(
                path,
                status="FAILED",
                locator="owner/private-g1a",
                failure_code=g1a.G1A_FAILURE_CODE,
            )
            self.assertIsNone(payload["g1a_seal_sha256"])
            self.assertFalse(payload["science_authorized"])
            self.assertFalse(payload["protected_test_accessed"])
            self.assertFalse(payload["external_surface_accessed"])

    def test_g1a_worker_fails_fast_on_current_runtime_failure(self):
        with tempfile.TemporaryDirectory() as td:
            runtime_sha = "d" * 40
            source = Path(td) / "source.json"
            source.write_text(
                json.dumps({
                    "schema_version": g1a.HANDOFF_SCHEMA_VERSION,
                    "stage": "TRACKA_V12_G1A_SHARED_HANDOFF",
                    "status": "FAILED",
                    "science_source_sha": SCIENCE_SHA,
                    "operator_runtime_sha": runtime_sha,
                    "private_kaggle_dataset_locator": "owner/private-g1a",
                    "g1a_seal_sha256": None,
                    "failure_code": g1a.G1A_FAILURE_CODE,
                    "science_authorized": False,
                    "protected_test_accessed": False,
                    "external_surface_accessed": False,
                }),
                encoding="utf-8",
            )

            def fake_fetch(repo, run_id, filename, destination):
                destination = Path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                return destination

            with mock.patch.object(g1a, "operator_runtime_head", return_value=runtime_sha), \
                 mock.patch.object(g1a, "fetch_public_file", side_effect=fake_fetch):
                with self.assertRaisesRegex(OperatorError, "K1 reported canonical G1A failure"):
                    g1a._wait_for_current_g1a_handoff(Path(td), Path(td) / "dest.json")

    def test_g2a_collector_fails_fast_on_account_failure(self):
        failed = {
            "status": "FAILED",
            "failure_code": g2a.G2A_FAILURE_CODE,
        }
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(g2a, "fetch_existing_g2a", return_value=None), \
             mock.patch.object(g2a, "_fetch_current_account_status", return_value=failed):
            with self.assertRaisesRegex(OperatorError, "reported G2A qualification failure"):
                g2a._wait_for_calibration_or_failure(
                    Path(td),
                    calibration_id="CAL-R13",
                    g1a_sha="e" * 64,
                    destination=Path(td) / "CAL-R13_SUMMARY.json",
                )

    def test_guard_reuses_terminal_result_without_rerun(self):
        with tempfile.TemporaryDirectory() as td:
            runtime_sha = "a" * 40
            root = Path(td)
            status = root / "K1.status.json"
            status.write_text(
                json.dumps({
                    "schema_version": guard.STATUS_SCHEMA,
                    "state": "FINISHED",
                    "account_id": "K1",
                    "science_sha": SCIENCE_SHA,
                    "operator_runtime_sha": runtime_sha,
                    "launch_id": "prior-launch",
                    "return_code": 0,
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"CROPCOP_MASTER_GUARD_ROOT": td}, clear=False), \
                 mock.patch.object(guard, "operator_runtime_head", return_value=runtime_sha), \
                 mock.patch.object(guard, "_run_owner") as run_owner:
                self.assertEqual(guard.launch_or_follow("K1"), 0)
                run_owner.assert_not_called()

    def test_guard_rejects_stale_running_marker(self):
        with tempfile.TemporaryDirectory() as td:
            runtime_sha = "b" * 40
            root = Path(td)
            status = root / "K1.status.json"
            status.write_text(
                json.dumps({
                    "schema_version": guard.STATUS_SCHEMA,
                    "state": "RUNNING",
                    "account_id": "K1",
                    "science_sha": SCIENCE_SHA,
                    "operator_runtime_sha": runtime_sha,
                    "launch_id": "stale-launch",
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"CROPCOP_MASTER_GUARD_ROOT": td}, clear=False), \
                 mock.patch.object(guard, "operator_runtime_head", return_value=runtime_sha):
                with self.assertRaises(OperatorError):
                    guard.launch_or_follow("K1")


if __name__ == "__main__":
    unittest.main()
