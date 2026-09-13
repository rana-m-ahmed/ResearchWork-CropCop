from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_analysis import ArchitectureSelectorRow, FAMILIES, ROBUSTNESS_CORRUPTIONS
from cropcop_je.tracka_v12_evidence import (
    ALL_DIRECT_STATES,
    NUM_CLASSES,
    build_tracka_selection_closure,
    canonical_prediction_digest,
    deterministic_subset,
    robustness_seed,
    summarize_prediction_rows,
    validate_efficiency_identity,
    validate_selected_checkpoint_replay,
    xai_random_control_seed,
    xai_sample,
)


def prediction_rows(count=120):
    rows = []
    for index in range(count):
        target = index % NUM_CLASSES
        rows.append(
            {
                "stable_row_id": f"row-{index:05d}",
                "target_class_index": target,
                "predicted_class_index": target,
                "true_class_nll": 0.1,
                "top1_confidence": 0.9,
            }
        )
    return rows


class TrackAV12EvidenceTests(unittest.TestCase):
    def test_prediction_summary_is_deterministic_and_exact(self):
        rows = prediction_rows()
        summary = summarize_prediction_rows(rows, expected_count=120)
        self.assertEqual(summary["row_count"], 120)
        self.assertEqual(summary["unique_row_id_count"], 120)
        self.assertEqual(summary["validation_accuracy"], 1.0)
        self.assertEqual(summary["validation_balanced_accuracy"], 1.0)
        self.assertEqual(summary["validation_macro_f1"], 1.0)
        self.assertAlmostEqual(summary["validation_nll"], 0.1)
        self.assertEqual(len(summary["classwise"]), 120)
        self.assertEqual(len(summary["class_f1"]), 120)
        self.assertEqual(summary["predictions_sha256"], canonical_prediction_digest(reversed(rows)))

    def test_duplicate_prediction_row_fails_closed(self):
        rows = prediction_rows()
        rows[-1]["stable_row_id"] = rows[0]["stable_row_id"]
        with self.assertRaises(ValueError):
            summarize_prediction_rows(rows, expected_count=120)

    def test_replay_gate_enforces_one_e_minus_six(self):
        summary = summarize_prediction_rows(prediction_rows(), expected_count=120)
        expected = {name: summary[name] for name in (
            "validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1", "validation_nll"
        )}
        self.assertEqual(validate_selected_checkpoint_replay(summary, expected)["status"], "PASS")
        bad = dict(expected)
        bad["validation_macro_f1"] -= 2e-6
        self.assertEqual(validate_selected_checkpoint_replay(summary, bad)["status"], "FAIL")

    def test_robustness_seed_is_exact_and_protocol_bounded(self):
        first = robustness_seed("row-1", "gaussian_noise_uint8", "2")
        self.assertEqual(first, robustness_seed("row-1", "gaussian_noise_uint8", "2"))
        self.assertNotEqual(first, robustness_seed("row-2", "gaussian_noise_uint8", "2"))
        with self.assertRaises(ValueError):
            robustness_seed("row-1", "gaussian_noise", "2")
        self.assertIn("gaussian_noise_uint8", ROBUSTNESS_CORRUPTIONS)

    def test_xai_sample_is_exact_two_per_class_and_prediction_blind(self):
        rows = []
        for class_index in range(NUM_CLASSES):
            for local in range(4):
                rows.append({"class_index": class_index, "stable_row_id": f"c{class_index:03d}-{local}"})
        sample = xai_sample(rows)
        self.assertEqual(len(sample), 240)
        self.assertEqual(len(set(sample)), 240)
        counts = {class_index: 0 for class_index in range(NUM_CLASSES)}
        lookup = {row["stable_row_id"]: row["class_index"] for row in rows}
        for row_id in sample:
            counts[lookup[row_id]] += 1
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(sample, xai_sample(list(reversed(rows))))

    def test_xai_subsets_and_random_controls_are_deterministic(self):
        row_ids = [f"r-{i}" for i in range(240)]
        self.assertEqual(
            deterministic_subset(row_ids, count=30, salt="sanity"),
            deterministic_subset(reversed(row_ids), count=30, salt="sanity"),
        )
        self.assertEqual(
            xai_random_control_seed("r-1", "model-1", 0.2, 3),
            xai_random_control_seed("r-1", "model-1", 0.2, 3),
        )
        with self.assertRaises(ValueError):
            xai_random_control_seed("r-1", "model-1", 0.25, 3)

    def test_efficiency_identity_must_match_all_three_seeds(self):
        row = {
            seed: {
                "model_state_tensor_bytes_fp32": 100,
                "total_parameter_count": 25,
                "trainable_parameter_count": 25,
                "input_resolution": 256,
            }
            for seed in ("S1", "S2", "S3")
        }
        self.assertEqual(validate_efficiency_identity(row)["model_state_tensor_bytes_fp32"], 100)
        row["S3"]["total_parameter_count"] = 26
        with self.assertRaises(ValueError):
            validate_efficiency_identity(row)

    def test_final_closure_requires_all_twelve_direct_states_and_all_gates(self):
        state = {
            experiment_id: {
                "terminal": True,
                "selected_checkpoint_verified": True,
                "replay_pass": True,
                "robustness_pass": True,
                "classwise_pass": True,
                "efficiency_pass": True,
                "v1_test_accessed": False,
                "external_surface_accessed": False,
            }
            for experiment_id in ALL_DIRECT_STATES
        }
        xai = {
            experiment_id: {"status": "PASS", "training_or_adaptation_performed": False}
            for experiment_id in ALL_DIRECT_STATES
        }
        rows = [
            ArchitectureSelectorRow(family, .95 - i * .01, .94 - i * .01, .80, 2.0, 100 + i, 20 + i)
            for i, family in enumerate(FAMILIES)
        ]
        closure = build_tracka_selection_closure(selector_rows=rows, state_gates=state, xai_gates=xai)
        self.assertEqual(closure["status"], "PASS")
        self.assertTrue(closure["science_selection_sealed"])
        self.assertFalse(closure["xai_used_as_weighted_selector"])
        self.assertFalse(closure["v1_test_accessed"])
        broken = dict(state)
        broken[ALL_DIRECT_STATES[0]] = dict(broken[ALL_DIRECT_STATES[0]])
        broken[ALL_DIRECT_STATES[0]]["replay_pass"] = False
        self.assertEqual(
            build_tracka_selection_closure(selector_rows=rows, state_gates=broken, xai_gates=xai)["status"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
