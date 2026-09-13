from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_historical import (  # noqa: E402
    HISTORICAL_SECONDARY_DIRECT_SPECS,
    HISTORICAL_TRACKA_CLOSURE_SPECS,
    validate_historical_closure_identity,
)


class TrackAV12HistoricalLineageTests(unittest.TestCase):
    def test_registry_contains_exact_ten_completed_states(self):
        expected = {
            "R04-MNV4-DIRECT-S1", "R04-MNV4-DIRECT-S2", "R04-MNV4-DIRECT-S3",
            "R05-MNV4-TEACHER-S1", "R05-MNV4-TEACHER-S2", "R05-MNV4-TEACHER-S3",
            "R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1",
            "R06-EFFB0-CONTEXT-S1", "R07-CNXTT-CONTEXT-S1",
        }
        self.assertEqual(set(HISTORICAL_TRACKA_CLOSURE_SPECS), expected)
        for experiment_id, row in HISTORICAL_TRACKA_CLOSURE_SPECS.items():
            self.assertEqual(len(row["source_git_commit"]), 40, experiment_id)
            self.assertEqual(len(row["selected_checkpoint_sha256"]), 64, experiment_id)
            self.assertGreater(int(row["selected_epoch"]), 0, experiment_id)

    def test_registry_matches_canonical_wave_closure_files(self):
        evidence = ROOT / "journal_extension" / "evidence" / "public" / "track_a"
        wave1 = json.loads((evidence / "WAVE1_PRINCIPAL_VALIDATION_CLOSURE.json").read_text(encoding="utf-8"))
        wave2 = json.loads((evidence / "WAVE2_SECONDARY_VALIDATION_CLOSURE.json").read_text(encoding="utf-8"))
        observed = {}
        for payload, wave in ((wave1, "WAVE1"), (wave2, "WAVE2")):
            for row in payload["runs"]:
                observed[row["experiment_id"]] = {
                    "wave": wave,
                    "run_id": row["run_id"],
                    "selected_epoch": row["selected_epoch"],
                    "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
                }
        self.assertEqual(set(observed), set(HISTORICAL_TRACKA_CLOSURE_SPECS))
        for experiment_id, canonical in HISTORICAL_TRACKA_CLOSURE_SPECS.items():
            self.assertEqual(observed[experiment_id]["wave"], canonical["wave"])
            self.assertEqual(observed[experiment_id]["run_id"], canonical["run_id"])
            self.assertEqual(observed[experiment_id]["selected_epoch"], canonical["selected_epoch"])
            self.assertEqual(observed[experiment_id]["selected_checkpoint_sha256"], canonical["selected_checkpoint_sha256"])

    def test_historical_identity_rejects_source_run_and_checkpoint_tampering(self):
        experiment_id = "R04-MNV4-DIRECT-S1"
        spec = HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]
        record = {
            "experiment_id": experiment_id,
            "status": "PASS",
            "run_id": spec["run_id"],
            "source_git_commit": spec["source_git_commit"],
            "continuation_required": False,
            "v1_test_accessed": False,
            "protected_external_surface_accessed": False,
            "artifact_locators": {"selected_checkpoint": {"sha256": spec["selected_checkpoint_sha256"]}},
            "result_summary": {
                "selected_checkpoint_sha256": spec["selected_checkpoint_sha256"],
                "selected_epoch": spec["selected_epoch"],
            },
        }
        self.assertEqual(validate_historical_closure_identity(record), [])
        for field, bad_value, expected_fragment in (
            ("source_git_commit", "f" * 40, "source_git_commit"),
            ("run_id", "wrong-run", "run_id"),
        ):
            broken = json.loads(json.dumps(record))
            broken[field] = bad_value
            self.assertTrue(any(expected_fragment in error for error in validate_historical_closure_identity(broken)))
        broken = json.loads(json.dumps(record))
        broken["artifact_locators"]["selected_checkpoint"]["sha256"] = "0" * 64
        broken["result_summary"]["selected_checkpoint_sha256"] = "0" * 64
        self.assertIn("historical selected checkpoint differs from immutable Wave closure", validate_historical_closure_identity(broken))

    def test_secondary_direct_specs_agree_with_canonical_closure_registry(self):
        for experiment_id, spec in HISTORICAL_SECONDARY_DIRECT_SPECS.items():
            self.assertEqual(spec["selected_checkpoint_sha256"], HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]["selected_checkpoint_sha256"])
            self.assertEqual(spec["run_id"], HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]["run_id"])


if __name__ == "__main__":
    unittest.main()
