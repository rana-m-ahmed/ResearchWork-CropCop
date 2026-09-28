from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class TrackCPublicEvidenceTest(unittest.TestCase):
    def test_final_public_evidence_validator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (
            root
            / "journal_extension"
            / "track_c_r07"
            / "analysis"
            / "validate_trackc_evidence.py"
        )
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        self.assertIn("PASS_TRACKC_PUBLIC_EVIDENCE_VALIDATION", proc.stdout)

    def test_poco_compatibility_validator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "journal_extension" / "track_c_r07" / "analysis" / "validate_trackc_poco_compatibility.py"
        proc = subprocess.run(
            [sys.executable, str(script)], cwd=root, text=True, capture_output=True, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        self.assertIn("PASS_TRACKC_POCO_COMPATIBILITY_VALIDATION", proc.stdout)

    def test_poco_frozen_runtime_blocker_validator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "journal_extension" / "track_c_r07" / "analysis" / "validate_trackc_secondary_device.py"
        proc = subprocess.run(
            [sys.executable, str(script)], cwd=root, text=True, capture_output=True, check=False
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        self.assertIn("PASS_TRACKC_SECONDARY_DEVICE_BLOCKER_VALIDATION", proc.stdout)


if __name__ == "__main__":
    unittest.main()
