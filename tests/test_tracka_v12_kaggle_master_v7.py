from __future__ import annotations

import importlib.util
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

import master_launch_guard_v7 as guard  # noqa: E402
from tracka_v12_kaggle_operator_v3 import SCIENCE_SHA, OperatorError  # noqa: E402


class TrackAV12KaggleMasterV7Tests(unittest.TestCase):
    def test_repaired_science_head_is_bound(self):
        self.assertEqual(SCIENCE_SHA, "e21a505792ddd91df55712e248391c79c55cf235")

    def test_driver_uses_direct_g1a_sealer_path(self):
        text = (OPS / "master_account_driver_v7.py").read_text(encoding="utf-8")
        self.assertIn("from master_g1a import", text)
        self.assertNotIn("master_g1a_v6", text)
        self.assertNotIn("master_g1a_sealer_compat_v6", text)
        self.assertNotIn("init_globals", text)
        self.assertNotIn("runpy", text)

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
