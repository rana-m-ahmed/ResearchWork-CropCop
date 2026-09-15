from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import master_g1a_v6 as g1a_v6
import master_singleton_v6 as singleton


class TrackAV12MasterV6Tests(unittest.TestCase):
    def test_frozen_sealer_defect_is_explicitly_bridged_without_source_edit(self):
        sealer = (ROOT / "journal_extension" / "scripts" / "seal_tracka_v12_g1a.py").read_text(encoding="utf-8")
        secondary = (SRC / "cropcop_je" / "secondary.py").read_text(encoding="utf-8")
        self.assertIn('"torchvision_version": TORCHVISION_VERSION', sealer)
        self.assertIn('TORCHVISION_VERSION = "0.27.1"', secondary)
        launcher = g1a_v6._compat_launcher_source()
        self.assertIn("from cropcop_je.secondary import TORCHVISION_VERSION", launcher)
        self.assertIn("init_globals={'TORCHVISION_VERSION': TORCHVISION_VERSION}", launcher)
        self.assertNotIn("write_text", launcher)

    def test_frozen_sealer_command_uses_runpy_compat_not_direct_script_execution(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            script = repo / "journal_extension" / "scripts" / "seal_tracka_v12_g1a.py"
            script.parent.mkdir(parents=True)
            script.write_text("raise SystemExit(0)\n", encoding="utf-8")
            cmd = g1a_v6.frozen_sealer_command(repo, ["--x", "y"])
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1], "-c")
            self.assertIn(str(script), cmd)
            self.assertEqual(cmd[-2:], ["--x", "y"])

    def test_singleton_terminal_marker_prevents_second_execution_in_same_session(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"CROPCOP_MASTER_SINGLETON_ROOT": td}, clear=False):
                lease = singleton.acquire_master_execution("K1", wait_seconds=1)
                self.assertTrue(lease.is_primary)
                lease.finish(0, "ok")
                again = singleton.acquire_master_execution("K1", wait_seconds=1)
                self.assertFalse(again.is_primary)
                self.assertEqual(again.mirror_code, 0)

    def test_concurrent_duplicate_waits_and_mirrors_primary_result(self):
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ)
            env["CROPCOP_MASTER_SINGLETON_ROOT"] = td
            env["PYTHONPATH"] = os.pathsep.join([str(OPS), str(SRC), env.get("PYTHONPATH", "")])
            with mock.patch.dict(os.environ, {"CROPCOP_MASTER_SINGLETON_ROOT": td}, clear=False):
                primary = singleton.acquire_master_execution("K2", wait_seconds=5)
                code = (
                    "from master_singleton_v6 import acquire_master_execution; "
                    "x=acquire_master_execution('K2', wait_seconds=5); "
                    "print('MIRROR_CODE='+str(x.mirror_code))"
                )
                child = subprocess.Popen(
                    [sys.executable, "-c", code],
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(0.5)
                primary.finish(2, "controlled")
                out, err = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, msg=err)
            self.assertIn("DUPLICATE_MASTER_INVOCATION K2", out)
            self.assertIn("MIRROR_CODE=2", out)

    def test_v6_driver_acquires_singleton_before_any_stage(self):
        source = (OPS / "master_account_driver_v6.py").read_text(encoding="utf-8")
        self.assertLess(source.index("acquire_master_execution(account_id)"), source.index("run_account(account_id)"))
        self.assertIn("from master_g1a_v6 import", source)
        self.assertNotIn("from master_g1a import acquire_canonical_g1a_worker, ensure_canonical_g1a_k1", source)

    def test_failed_primary_is_terminal_and_not_automatically_reexecuted(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"CROPCOP_MASTER_SINGLETON_ROOT": td}, clear=False):
                first = singleton.acquire_master_execution("K3", wait_seconds=1)
                first.finish(1, "hard failure")
                second = singleton.acquire_master_execution("K3", wait_seconds=1)
                self.assertEqual(second.mirror_code, 1)
                marker = json.loads((Path(td) / ".cropcop-tracka-master-K3.json").read_text())
                self.assertEqual(marker["status"], "FAILED")
                self.assertEqual(marker["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
