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


if __name__ == "__main__":
    unittest.main()
