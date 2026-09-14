from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    sys.path.insert(0, str(path))

import master_account_driver_v6 as driver_v6
import master_g1a_sealer_compat_v6 as compat_v6
import master_g1a_v6
import master_launch_guard_v6 as guard_v6


class TrackAV12MasterV6Tests(unittest.TestCase):
    def test_frozen_sealer_has_exact_known_missing_global_shape(self):
        script = ROOT / compat_v6.SEALER_RELATIVE_PATH
        blob = subprocess.check_output(["git", "hash-object", str(script)], cwd=ROOT, text=True).strip()
        self.assertEqual(blob, compat_v6.EXPECTED_FROZEN_SEALER_GIT_BLOB)
        tree = ast.parse(script.read_text(encoding="utf-8"))
        bound = compat_v6._bound_module_names(tree)
        loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        self.assertIn("TORCHVISION_VERSION", loads)
        self.assertNotIn("TORCHVISION_VERSION", bound)

    def test_frozen_secondary_is_single_source_for_injected_version(self):
        from cropcop_je.secondary import TORCHVISION_VERSION
        self.assertEqual(TORCHVISION_VERSION, "0.27.1")
        self.assertEqual(TORCHVISION_VERSION, compat_v6.EXPECTED_TORCHVISION_VERSION)

    def test_compatibility_contract_accepts_exact_frozen_tree(self):
        contract = compat_v6.validate_frozen_sealer_contract(ROOT)
        self.assertEqual(contract["sealer_git_blob"], compat_v6.EXPECTED_FROZEN_SEALER_GIT_BLOB)
        self.assertEqual(contract["injected_global"], "TORCHVISION_VERSION")
        self.assertEqual(contract["injected_value"], "0.27.1")

    def test_v6_g1a_routes_only_through_compatibility_runner(self):
        source = (OPS / "master_g1a_v6.py").read_text(encoding="utf-8")
        self.assertIn("master_g1a_sealer_compat_v6.py", source)
        self.assertIn("build_g1a_once_v6", source)
        self.assertNotIn('str(repo / "journal_extension/scripts/seal_tracka_v12_g1a.py")', source)
        self.assertIs(master_g1a_v6.ensure_canonical_g1a_k1, master_g1a_v6._base.ensure_canonical_g1a_k1)
        self.assertIs(master_g1a_v6._base.build_g1a_once, master_g1a_v6.build_g1a_once_v6)

    def test_v6_driver_refuses_direct_unserialized_execution(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception):
                driver_v6.guard_identity("K1")
        with mock.patch.dict(os.environ, {
            "CROPCOP_MASTER_GUARD_TOKEN": "0123456789abcdef0123456789abcdef",
            "CROPCOP_MASTER_GUARD_ACCOUNT": "K1",
            "CROPCOP_MASTER_GUARD_LOCK": "/tmp/x.lock",
        }, clear=True):
            token = driver_v6.guard_identity("K1")
            self.assertEqual(token, "0123456789abcdef0123456789abcdef")

    def test_launch_guard_terminal_status_must_match_runtime_account_and_time(self):
        base = {
            "schema_version": guard_v6.STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": "K1",
            "science_sha": guard_v6.SCIENCE_SHA,
            "operator_runtime_sha": "runtime",
            "finished_unix": 101.0,
            "return_code": 0,
        }
        self.assertTrue(guard_v6._status_matches_recent_owner(base, account_id="K1", runtime_sha="runtime", wait_started_unix=100.0))
        for key, value in (
            ("account_id", "K2"),
            ("operator_runtime_sha", "other"),
            ("state", "RUNNING"),
            ("finished_unix", 90.0),
        ):
            broken = dict(base)
            broken[key] = value
            self.assertFalse(guard_v6._status_matches_recent_owner(broken, account_id="K1", runtime_sha="runtime", wait_started_unix=100.0))

    def test_launch_guard_owner_writes_terminal_status_and_propagates_rc(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = root / "K1.lock"
            status = root / "K1.status.json"
            cp = subprocess.CompletedProcess(args=["python"], returncode=2)
            with mock.patch.object(guard_v6.subprocess, "run", return_value=cp), \
                 mock.patch.object(guard_v6, "operator_runtime_head", return_value="runtime-sha"):
                rc = guard_v6._run_owner("K1", lock_path=lock, status_path=status, runtime_sha="runtime-sha")
            self.assertEqual(rc, 2)
            payload = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "FINISHED")
            self.assertEqual(payload["return_code"], 2)
            self.assertEqual(payload["account_id"], "K1")
            self.assertEqual(payload["operator_runtime_sha"], "runtime-sha")

    def test_v6_stage_markers_include_pid_and_launch_identity(self):
        with mock.patch("builtins.print") as printer:
            driver_v6.stage("CANONICAL_G1A", "K1", "abcdef0123456789")
        rendered = " ".join(str(x) for call in printer.call_args_list for x in call.args)
        self.assertIn("CANONICAL_G1A", rendered)
        self.assertIn("pid=", rendered)
        self.assertIn("launch=abcdef012345", rendered)

    def test_v6_driver_keeps_protected_surfaces_closed(self):
        combined = "\n".join((OPS / name).read_text(encoding="utf-8") for name in (
            "master_account_driver_v6.py",
            "master_g1a_v6.py",
            "master_launch_guard_v6.py",
        ))
        self.assertNotIn("DS-V1-TEST-CONSUMED", combined)
        self.assertNotIn("track_b", combined.lower())
        self.assertNotIn("track_c", combined.lower())


if __name__ == "__main__":
    unittest.main()
