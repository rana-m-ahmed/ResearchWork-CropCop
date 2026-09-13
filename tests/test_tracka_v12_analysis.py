from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_analysis import (  # noqa: E402
    ArchitectureSelectorRow,
    ROBUSTNESS_CORRUPTIONS,
    class_tail_summary,
    corruption_summary,
    dominates,
    pareto_frontier,
    q8,
    select_journal_primary,
)


def row(family, mean, worst, tail, degradation, size, params):
    return ArchitectureSelectorRow(
        family=family,
        mean_validation_macro_f1=mean,
        worst_seed_validation_macro_f1=worst,
        bottom_12_class_mean_f1=tail,
        mean_corruption_degradation_pp=degradation,
        model_state_tensor_bytes_fp32=size,
        total_parameter_count=params,
    )


class TrackAV12AnalysisTests(unittest.TestCase):
    def test_selector_precision_is_half_even_eight_places(self):
        self.assertEqual(str(q8("0.123456785")), "0.12345678")
        self.assertEqual(str(q8("0.123456795")), "0.12345680")

    def test_bottom_12_is_deterministic_and_class_index_breaks_ties(self):
        vectors = {label: [0.9] * 120 for label in ("S1", "S2", "S3")}
        for class_index in range(20):
            for label in vectors:
                vectors[label][class_index] = 0.2
        summary = class_tail_summary(vectors)
        self.assertEqual(summary["bottom_class_indices"], list(range(12)))
        self.assertAlmostEqual(summary["bottom_12_class_mean_f1"], 0.2)
        self.assertEqual(summary["minimum_class_index"], 0)

    def test_robustness_inventory_matches_frozen_protocol(self):
        self.assertEqual(
            ROBUSTNESS_CORRUPTIONS,
            ("brightness", "contrast", "gaussian_blur", "gaussian_noise_uint8", "jpeg"),
        )

    def test_corruption_degradation_retains_negative_values(self):
        clean = {label: 0.8 for label in ("S1", "S2", "S3")}
        corruptions = {}
        for label in clean:
            corruptions[label] = {}
            for corruption in ROBUSTNESS_CORRUPTIONS:
                corruptions[label][corruption] = {"1": 0.81, "2": 0.79, "3": 0.78}
        summary = corruption_summary(clean, corruptions)
        self.assertAlmostEqual(summary["mean_corruption_degradation_pp"], (-1.0 + 1.0 + 2.0) / 3.0)
        self.assertEqual(summary["severity_monotonicity_violations_count"], 0)

    def test_legacy_gaussian_noise_alias_is_rejected(self):
        clean = {label: 0.8 for label in ("S1", "S2", "S3")}
        corruptions = {}
        for label in clean:
            corruptions[label] = {}
            for corruption in ("brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg"):
                corruptions[label][corruption] = {"1": 0.8, "2": 0.8, "3": 0.8}
        with self.assertRaisesRegex(ValueError, "corruption inventory mismatch"):
            corruption_summary(clean, corruptions)

    def test_pareto_dominance_uses_all_five_dimensions(self):
        a = row("R04", .90, .89, .80, 3.0, 10, 10)
        b = row("R06", .89, .88, .79, 4.0, 11, 11)
        self.assertTrue(dominates(a, b))
        c = row("R07", .91, .90, .81, 2.0, 20, 20)
        self.assertFalse(dominates(a, c))

    def test_single_nondominated_selects_at_pareto_stage(self):
        rows = [
            row("R04", .95, .94, .90, 1.0, 10, 10),
            row("R06", .94, .93, .89, 2.0, 11, 11),
            row("R07", .93, .92, .88, 3.0, 12, 12),
            row("R13", .92, .91, .87, 4.0, 13, 13),
        ]
        result = select_journal_primary(rows)
        self.assertEqual(result["selection_stage"], "pareto_single_nondominated")
        self.assertEqual(result["journal_primary_family"], "R04")

    def test_lexicographic_is_applied_only_on_frontier(self):
        rows = [
            row("R04", .95, .90, .80, 4.0, 10, 10),
            row("R06", .94, .93, .82, 3.0, 11, 11),
            row("R07", .93, .92, .81, 5.0, 20, 20),
            row("R13", .92, .91, .79, 6.0, 30, 30),
        ]
        frontier = {x.family for x in pareto_frontier(rows)}
        self.assertEqual(frontier, {"R04", "R06"})
        result = select_journal_primary(rows)
        self.assertEqual(result["selection_stage"], "frozen_lexicographic")
        self.assertEqual(result["journal_primary_family"], "R04")

    def test_exact_selector_precision_tie_reports_co_primary(self):
        rows = [
            row("R04", .95, .94, .90, 1.0, 10, 10),
            row("R06", .95, .94, .90, 1.0, 10, 10),
            row("R07", .90, .89, .80, 3.0, 20, 20),
            row("R13", .89, .88, .79, 4.0, 30, 30),
        ]
        result = select_journal_primary(rows)
        self.assertEqual(result["status"], "CO_PRIMARY_TIE")
        self.assertEqual(result["co_primary_families"], ["R04", "R06"])

    def test_selector_rejects_incomplete_candidate_pool(self):
        with self.assertRaises(ValueError):
            select_journal_primary([
                row("R04", .9, .9, .8, 1, 10, 10),
                row("R06", .9, .9, .8, 1, 10, 10),
                row("R07", .9, .9, .8, 1, 10, 10),
            ])


if __name__ == "__main__":
    unittest.main()
