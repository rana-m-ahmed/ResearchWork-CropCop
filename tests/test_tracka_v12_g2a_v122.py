from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12 import EXPERIMENT_SPECS  # noqa: E402
from cropcop_je.tracka_v12_g1a import R13_PRETRAINED_SHA256, TEACHER_SHA256  # noqa: E402
from cropcop_je.tracka_v12_g2a import PROFILE_COVERAGE, REQUIRED_PROFILES, SLOT_ORDER  # noqa: E402
from cropcop_je.tracka_v12_g2a_v122 import (  # noqa: E402
    CHECKPOINT_EVENT_UPPER_BOUND,
    STEP_ATTEMPTS_TOTAL,
    VAL_EXAMPLES_TOTAL,
    build_g2a_v122_barrier,
    build_scheduler_freeze_v122,
    profile_full_run_forecast,
    validate_calibration_summary_v122,
    validate_g2a_v122_barrier,
    validate_scheduler_freeze_v122,
)


def calibration(calibration_id: str, steady: float, val_rate: float, save: float) -> dict:
    row = {
        "schema_version": "1.2.2",
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
            "sec_per_optimizer_step": steady + 1.0,
            "steady_sec_per_optimizer_step": steady,
            "examples_per_second": 100.0,
            "validation_end_to_end_examples_per_second": val_rate,
            "checkpoint_save_seconds_per_event": save,
            "durable_sync_seconds_per_segment": 5.0,
            "peak_gpu_memory_bytes": 1,
            "checkpoint_save_seconds": save * 2,
            "checkpoint_load_seconds": 0.1,
        },
        "durability": {
            "sync_status": "PASS",
            "restore_status": "PASS",
            "sync_seconds": 1.0,
            "restore_seconds": 1.0,
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


class TrackAV12G2AV122Tests(unittest.TestCase):
    def summaries(self):
        costs = {
            "CAL-EFFB0": (1.0, 200.0, 0.2),
            "CAL-CNXTT": (2.5, 100.0, 0.5),
            "CAL-MNV4-LOGITS": (1.3, 180.0, 0.25),
            "CAL-MNV4-FEATURE": (1.8, 170.0, 0.25),
            "CAL-R13": (3.0, 90.0, 0.6),
        }
        return [calibration(cid, *costs[cid]) for cid in REQUIRED_PROFILES]

    def test_forecast_constants_match_frozen_30_epoch_contract(self):
        self.assertEqual(STEP_ATTEMPTS_TOTAL, 35820)
        self.assertEqual(VAL_EXAMPLES_TOTAL, 491040)
        self.assertEqual(CHECKPOINT_EVENT_UPPER_BOUND, 203)

    def test_valid_v122_calibrations_pass(self):
        for row in self.summaries():
            self.assertEqual(validate_calibration_summary_v122(row), [])

    def test_full_run_forecast_uses_training_validation_and_checkpoint_cost(self):
        row = calibration("CAL-R13", 2.0, 100.0, 0.5)
        forecast = profile_full_run_forecast(row)
        expected = 2.0 * STEP_ATTEMPTS_TOTAL + VAL_EXAMPLES_TOTAL / 100.0 + 0.5 * CHECKPOINT_EVENT_UPPER_BOUND + 5.0
        self.assertAlmostEqual(forecast["estimated_full_run_seconds_upper_bound"], expected)

    def test_missing_validation_throughput_fails_closed(self):
        row = calibration("CAL-R13", 2.0, 100.0, 0.5)
        del row["measured"]["validation_end_to_end_examples_per_second"]
        self.assertTrue(any("validation_end_to_end" in error for error in validate_calibration_summary_v122(row)))

    def test_barrier_covers_exact_11_states(self):
        barrier = build_g2a_v122_barrier(self.summaries(), checkpoint_contract_probe=checkpoint_probe())
        self.assertEqual(barrier["status"], "PASS")
        self.assertEqual(set(barrier["covered_experiment_ids"]), set(EXPERIMENT_SPECS))
        self.assertEqual(validate_g2a_v122_barrier(barrier), [])

    def test_scheduler_uses_full_run_cost_and_fills_six_slots(self):
        barrier = build_g2a_v122_barrier(self.summaries(), checkpoint_contract_probe=checkpoint_probe())
        freeze = build_scheduler_freeze_v122(barrier)
        self.assertEqual(
            validate_scheduler_freeze_v122(freeze, expected_g2a_barrier_sha256=barrier["barrier_sha256"]),
            [],
        )
        self.assertEqual(len(freeze["initial_dispatch"]), 6)
        self.assertEqual([row["slot"] for row in freeze["initial_dispatch"]], list(SLOT_ORDER))
        self.assertEqual(set(freeze["priority"]), set(EXPERIMENT_SPECS))
        self.assertEqual(freeze["priority"][:3], sorted(PROFILE_COVERAGE["CAL-R13"]))

    def test_physical_slot_dependence_is_rejected(self):
        barrier = build_g2a_v122_barrier(self.summaries(), checkpoint_contract_probe=checkpoint_probe())
        freeze = build_scheduler_freeze_v122(barrier)
        freeze["checkpoint_identity_is_physical_slot_independent"] = False
        self.assertTrue(validate_scheduler_freeze_v122(freeze))


if __name__ == "__main__":
    unittest.main()
