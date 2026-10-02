import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUN_LANE_PATH = ROOT / "journal_extension" / "kaggle" / "run_lane.py"

spec = importlib.util.spec_from_file_location("g1p_v2_run_lane", RUN_LANE_PATH)
run_lane = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = run_lane
spec.loader.exec_module(run_lane)


class G1PWave1ContractRegression(unittest.TestCase):
    def test_run_lane_g1_barrier_supplies_terminal_dual_smoke_evidence(self):
        """Executable regression for the audited G1->G2 barrier CLI contract.

        This intentionally exercises run_lane.validate_g1() and captures the
        actual validate_g1_barrier.py argv.  It must fail on the pre-fix source
        because the barrier parser requires --dual-gpu-smoke-evidence while
        run_lane omits it.
        """
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bundle = tmp / "g1"
            bundle.mkdir()
            (bundle / "G1_MODEL_IDENTITY_SEAL.json").write_text(
                json.dumps({"g1_seal_sha256": "a" * 64}) + "\n",
                encoding="utf-8",
            )
            output = tmp / "out"
            output.mkdir()

            env = {
                "CROPCOP_G1_BUNDLE_DIR": str(bundle),
                "CROPCOP_MANIFEST": str(tmp / "manifest.csv"),
                "CROPCOP_CLASS_MAP": str(tmp / "class_map.json"),
                "CROPCOP_MNV4_PRETRAINED": str(tmp / "mnv4.pt"),
                "CROPCOP_TEACHER_CHECKPOINT": str(tmp / "teacher.pt"),
                "CROPCOP_TEACHER_FACTORY_ROOT": str(tmp / "teacher_factory"),
                "CROPCOP_INFRA_SMOKE_EVIDENCE": str(tmp / "SMOKE_B_EVIDENCE.json"),
                "CROPCOP_DUAL_GPU_SMOKE_EVIDENCE": str(tmp / "DUAL_GPU_SMOKE_EVIDENCE.json"),
            }
            captured = []

            def fake_run(cmd, *, check=True):
                captured.append(list(cmd))
                if any(str(x).endswith("validate_g1_barrier.py") for x in cmd):
                    report = Path(cmd[cmd.index("--output") + 1])
                    report.write_text(
                        json.dumps({"status": "PASS"}) + "\n",
                        encoding="utf-8",
                    )
                return mock.Mock(returncode=0)

            with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
                run_lane, "run", side_effect=fake_run
            ):
                run_lane.validate_g1("f" * 40, "K1", output)

            barrier_calls = [
                cmd for cmd in captured
                if any(str(x).endswith("validate_g1_barrier.py") for x in cmd)
            ]
            self.assertEqual(len(barrier_calls), 1)
            argv = barrier_calls[0]
            self.assertIn(
                "--dual-gpu-smoke-evidence",
                argv,
                "run_lane.validate_g1() does not satisfy validate_g1_barrier.py's required dual-smoke contract",
            )
            idx = argv.index("--dual-gpu-smoke-evidence")
            self.assertEqual(
                argv[idx + 1],
                env["CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"],
            )


if __name__ == "__main__":
    unittest.main()
