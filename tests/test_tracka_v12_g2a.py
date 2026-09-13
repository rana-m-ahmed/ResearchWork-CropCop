from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12 import EXPERIMENT_SPECS  # noqa: E402
from cropcop_je.tracka_v12_g1a import R13_PRETRAINED_SHA256, TEACHER_SHA256  # noqa: E402
from cropcop_je.tracka_v12_g2a import (  # noqa: E402
    PROFILE_COVERAGE,
    REQUIRED_PROFILES,
    SLOT_ORDER,
    build_g2a_barrier,
    build_scheduler_freeze,
    validate_calibration_summary,
    validate_g2a_barrier_object,
    validate_scheduler_freeze,
)


def calibration(calibration_id: str, cost: float) -> dict:
    row = {
        "calibration_id": calibration_id,
        "status": "PASS",
        "calibration_weights_scientific": False,
        "validation_enabled": False,
        "scientific_metric_computed": False,
        "resume_success": True,
        "durable_roundtrip_success": True,
        "visible_cuda_device_count": 1,
        "visible_gpu_name": "NVIDIA T4",
        "git_credentials_present": False,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
        "coverage": sorted(PROFILE_COVERAGE[calibration_id]),
        "source_git_commit": "1" * 40,
        "software_stack_sha256": "2" * 64,
        "g1a_seal_sha256": "3" * 64,
        "dependency_lock_sha256": "4" * 64,
        "measured": {
            "sec_per_optimizer_step": cost,
            "examples_per_second": 100.0,
            "peak_gpu_memory_bytes": 1,
            "checkpoint_save_seconds": 0.1,
            "checkpoint_load_seconds": 0.1,
        },
        "durability": {
            "sync_status": "PASS",
            "restore_status": "PASS",
            "sync_seconds": 0.1,
            "restore_seconds": 0.1,
        },
    }
    if calibration_id.startswith("CAL-MNV4-"):
        row["teacher_checkpoint_sha256"] = TEACHER_SHA256
    if calibration_id == "CAL-R13":
        row["r13_pretrained_sha256"] = R13_PRETRAINED_SHA256
    return row


def checkpoint_probe() -> dict:
    return {
        "status": "PASS",
        "selected_checkpoint_verified": True,
        "identity_mismatch_rejected": True,
        "corrupt_checkpoint_rejected": True,
    }


class TrackAV12G2ATests(unittest.TestCase):
    def summaries(self):
        costs = {
            "CAL-EFFB0": 1.0,
            "CAL-CNXTT": 3.0,
            "CAL-MNV4-LOGITS": 1.5,
            "CAL-MNV4-FEATURE": 2.0,
            "CAL-R13": 4.0,
        }
        return [calibration(cid, costs[cid]) for cid in REQUIRED_PROFILES]

    def test_each_valid_calibration_passes(self):
        for row in self.summaries():
            self.assertEqual(validate_calibration_summary(row), [])

    def test_g2a_barrier_covers_exact_11_states(self):
        barrier = build_g2a_barrier(self.summaries(), checkpoint_contract_probe=checkpoint_probe())
        self.assertEqual(barrier["status"], "PASS")
        self.assertEqual(set(barrier["covered_experiment_ids"]), set(EXPERIMENT_SPECS))
        self.assertEqual(validate_g2a_barrier_object(barrier), [])

    def test_validation_metrics_are_forbidden_in_calibration(self):
        row = calibration("CAL-R13", 4.0)
        row["validation_enabled"] = True
        row["scientific_metric_computed"] = True
        errors = validate_calibration_summary(row)
        self.assertTrue(any("validation" in x for x in errors))
        self.assertTrue(any("scientific metric" in x for x in errors))

    def test_wrong_gpu_visibility_fails(self):
        row = calibration("CAL-EFFB0", 1.0)
        row["visible_cuda_device_count"] = 2
        self.assertTrue(validate_calibration_summary(row))

    def test_incomplete_profile_set_fails_barrier(self):
        barrier = build_g2a_barrier(self.summaries()[:-1], checkpoint_contract_probe=checkpoint_probe())
        self.assertEqual(barrier["status"], "FAIL")
        self.assertTrue(barrier["errors"])

    def test_scheduler_is_lpt_and_fills_six_slots(self):
        barrier = build_g2a_barrier(self.summaries(), checkpoint_contract_probe=checkpoint_probe())
        freeze = build_scheduler_freeze(barrier)
        self.assertEqual(validate_scheduler_freeze(freeze, expected_g2a_barrier_sha256=barrier["barrier_sha256"]), [])
        self.assertEqual(len(freeze["initial_dispatch"]), 6)
        self.assertEqual([x["slot"] for x in freeze["initial_dispatch"]], list(SLOT_ORDER))
        r13_positions = [freeze["priority"].index(eid) for eid in PROFILE_COVERAGE["CAL-R13"]]
        self.assertEqual(sorted(r13_positions), [0, 1, 2])
        self.assertEqual(set(freeze["priority"]), set(EXPERIMENT_SPECS))

    def test_lpt_equal_cost_tie_breaks_by_experiment_id(self):
        summaries = [calibration(cid, 1.0) for cid in REQUIRED_PROFILES]
        barrier = build_g2a_barrier(summaries, checkpoint_contract_probe=checkpoint_probe())
        freeze = build_scheduler_freeze(barrier)
        self.assertEqual(freeze["priority"], sorted(EXPERIMENT_SPECS))

    def test_scheduler_hash_drift_is_rejected(self):
        barrier = build_g2a_barrier(self.summaries(), checkpoint_contract_probe=checkpoint_probe())
        freeze = build_scheduler_freeze(barrier)
        freeze["priority"] = list(reversed(freeze["priority"]))
        self.assertTrue(any("self-hash" in x for x in validate_scheduler_freeze(freeze)))


if __name__ == "__main__":
    unittest.main()
