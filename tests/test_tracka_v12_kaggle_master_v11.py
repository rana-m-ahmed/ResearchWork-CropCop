from __future__ import annotations

import json
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

import tracka_v12_kaggle_operator_v8 as active
import master_continuation_v11 as cont
import master_account_driver_v11 as driver
import master_launch_guard_v11 as guard
import master_g1a_v8 as g1a

SCIENCE_SHA = "56023042e57758591df9babb3438f191dbe10312"


class TrackAV12KaggleMasterV11Tests(unittest.TestCase):
    def _write_handoff(self, path: Path, **updates) -> Path:
        payload = {
            "schema_version": "1.2.1",
            "stage": "TRACKA_V12_G1A_SHARED_HANDOFF",
            "status": "READY",
            "science_source_sha": SCIENCE_SHA,
            "operator_runtime_sha": "r" * 40,
            "private_kaggle_dataset_locator": "owner/cropcop-g1a",
            "g1a_seal_sha256": "a" * 64,
            "failure_code": None,
            "science_authorized": False,
            "protected_test_accessed": False,
            "external_surface_accessed": False,
        }
        payload.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_g1a_handoff_returns_none_without_polling(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(cont, "fetch_public_file", return_value=None):
            result = cont.probe_current_g1a_handoff(Path(td), destination=Path(td) / "handoff.json")
        self.assertIsNone(result)

    def test_stale_runtime_g1a_handoff_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_handoff(Path(td) / "handoff.json", operator_runtime_sha="old" * 10)
            with mock.patch.object(cont, "fetch_public_file", return_value=path), \
                 mock.patch.object(cont, "operator_runtime_head", return_value="r" * 40):
                result = cont.probe_current_g1a_handoff(Path(td), destination=path)
        self.assertIsNone(result)

    def test_exact_ready_g1a_handoff_passes_once(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_handoff(Path(td) / "handoff.json")
            with mock.patch.object(cont, "fetch_public_file", return_value=path), \
                 mock.patch.object(cont, "operator_runtime_head", return_value="r" * 40):
                result = cont.probe_current_g1a_handoff(Path(td), destination=path)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["g1a_seal_sha256"], "a" * 64)

    def test_exact_failed_g1a_handoff_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_handoff(
                Path(td) / "handoff.json",
                status="FAILED",
                g1a_seal_sha256=None,
                failure_code=g1a.G1A_FAILURE_CODE,
            )
            with mock.patch.object(cont, "fetch_public_file", return_value=path), \
                 mock.patch.object(cont, "operator_runtime_head", return_value="r" * 40):
                with self.assertRaises(active.OperatorError):
                    cont.probe_current_g1a_handoff(Path(td), destination=path)

    def test_worker_probe_precedes_expensive_stack_install(self):
        text = (OPS / "master_account_driver_v11.py").read_text(encoding="utf-8")
        probe = text.index('stage("WORKER_G1A_PREREQUISITE_PROBE"')
        stack = text.index('stage("EXACT_EXECUTION_STACK"')
        self.assertLess(probe, stack)
        self.assertIn("skipping expensive stack installation", text)
        self.assertIn("return controlled_dependency_continuation", text)

    def test_worker_private_access_is_checked_before_blocking_acquire(self):
        text = (OPS / "master_account_driver_v11.py").read_text(encoding="utf-8")
        access = text.index("if not kaggle_dataset_exists(expected_locator")
        acquire = text.index("g1a_bundle, g1a_seal, shared_locator = acquire_canonical_g1a_worker")
        self.assertLess(access, acquire)
        self.assertIn("Grant this Kaggle account Can view access", text)

    def test_v11_continuation_helper_has_no_poll_loop(self):
        text = (OPS / "master_continuation_v11.py").read_text(encoding="utf-8")
        self.assertNotIn("while True", text)
        self.assertNotIn("time.sleep", text)
        self.assertIn("probe_current_g1a_handoff", text)

    def test_driver_has_periodic_heartbeat(self):
        text = (OPS / "master_account_driver_v11.py").read_text(encoding="utf-8")
        self.assertIn("HEARTBEAT_SECONDS = 60.0", text)
        self.assertIn("MASTER_HEARTBEAT_V11", text)
        self.assertIn("flush=True", text)

    def test_controlled_continuation_is_rc2(self):
        self.assertEqual(driver.controlled_dependency_continuation("K2", "G1A not ready"), 2)

    def test_v11_guard_invokes_v11_driver_and_uses_fresh_status_root(self):
        text = (OPS / "master_launch_guard_v11.py").read_text(encoding="utf-8")
        self.assertIn("master_account_driver_v11.py", text)
        self.assertIn(".cropcop_tracka_master_guard_v11", text)
        status = {
            "schema_version": guard.STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": "K2",
            "science_sha": SCIENCE_SHA,
            "operator_runtime_sha": "r" * 40,
            "return_code": 2,
        }
        self.assertEqual(guard._matching_terminal_code(status, account_id="K2", runtime_sha="r" * 40), 2)

    def test_science_identity_and_protected_surfaces_unchanged(self):
        self.assertEqual(active.SCIENCE_SHA, SCIENCE_SHA)
        names = (
            "master_continuation_v11.py",
            "master_account_driver_v11.py",
            "master_launch_guard_v11.py",
        )
        text = "\n".join((OPS / name).read_text(encoding="utf-8") for name in names)
        for forbidden in ("DS-V1-TEST-CONSUMED", "DS-EXT-*-SEALED", "master_g1a_sealer_compat_v6"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
