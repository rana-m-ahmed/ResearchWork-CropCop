from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.runlog import claim_run_directory  # noqa: E402
from cropcop_je.train import _identity as checkpoint_identity  # noqa: E402
from run_tracka_v12_training_v121 import SLOT_IDS, logical_lane_id  # noqa: E402


class TrackAV12PlacementTests(unittest.TestCase):
    def test_logical_lane_is_experiment_stable(self):
        experiment_id = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
        logical = logical_lane_id(experiment_id)
        self.assertEqual(logical, "TRACKA-V12:R13-VIT-DLITTLE-DIFF-CONTEXT-S1")
        for _physical_slot in SLOT_IDS:
            self.assertEqual(logical_lane_id(experiment_id), logical)

    def test_checkpoint_identity_does_not_contain_physical_slot(self):
        experiment_id = "R13-VIT-DLITTLE-DIFF-CONTEXT-S2"
        logical = logical_lane_id(experiment_id)
        base = {
            "experiment_id": experiment_id,
            "authority_id": "EAAI-JE-SDL-v2.1-QA",
            "source_git_commit": "1" * 40,
            "config_sha256": "2" * 64,
            "ctc_v2_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "class_map_sha256": "5" * 64,
            "seed": 606135704,
            "student_init_sha256": "6" * 64,
            "pretrained_sha256": "7" * 64,
            "teacher_sha256": None,
            "teacher_factory_sha256": None,
            "teacher_factory_bundle_sha256": None,
            "software_stack_sha256": "8" * 64,
            "dependency_lock_sha256": "9" * 64,
            "g1_seal_sha256": "a" * 64,
            "g2_barrier_sha256": "b" * 64,
            "lane_id": logical,
            "physical_slot_id": "K1/GPU0",
        }
        first = checkpoint_identity(base)
        migrated = dict(base)
        migrated["physical_slot_id"] = "K3/GPU1"
        second = checkpoint_identity(migrated)
        self.assertEqual(first, second)
        self.assertEqual(first["lane_id"], logical)
        self.assertNotIn("physical_slot_id", first)

    def test_run_directory_owner_is_logical_not_physical(self):
        experiment_id = "R06-EFFB0-CONTEXT-S2"
        logical = logical_lane_id(experiment_id)
        with tempfile.TemporaryDirectory() as td:
            first = claim_run_directory(td, run_id="RUN-1", experiment_id=experiment_id, lane_id=logical)
            second = claim_run_directory(td, run_id="RUN-1", experiment_id=experiment_id, lane_id=logical)
            self.assertEqual(first, second)
            self.assertEqual(first["lane_id"], logical)

    def test_unauthorized_experiment_has_no_logical_lane(self):
        with self.assertRaises(ValueError):
            logical_lane_id("R99-UNAUTHORIZED-S1")


if __name__ == "__main__":
    unittest.main()
