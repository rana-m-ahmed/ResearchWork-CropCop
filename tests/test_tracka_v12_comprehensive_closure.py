from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import seal_tracka_v12_comprehensive_closure as closure  # noqa: E402

SOURCE = "a" * 40


def seal(payload):
    payload["closure_sha256"] = closure.sha256_json(payload)
    return payload


def fixtures(selection_status="SELECTED"):
    direct = seal({
        "status": "PASS",
        "science_selection_sealed": True,
        "closure_kind": "track_a_direct_model_selection",
        "closure_source_git_commit": SOURCE,
        "analysis_source_git_commit": SOURCE,
        "direct_state_inventory": sorted(closure.ALL_DIRECT_STATES),
        "track_b_handoff_authorized": False,
        "track_c_handoff_authorized": False,
        "v1_test_accessed": False,
        "external_predictions_opened": False,
        "track_c_candidate_runtime_opened": False,
        "selection": {
            "status": selection_status,
            "journal_primary_family": "R13" if selection_status == "SELECTED" else None,
            "co_primary_families": [] if selection_status == "SELECTED" else ["R06", "R13"],
        },
    })
    auxiliary = seal({
        "status": "PASS",
        "closure_kind": "track_a_auxiliary_three_seed_analysis",
        "closure_source_git_commit": SOURCE,
        "analysis_source_git_commit": SOURCE,
        "state_inventory": sorted(closure.R04 | closure.R05 | closure.R12),
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "hypothesis_tests_authorized": False,
        "multiple_comparison_p_values_authorized": False,
        "paired_analyses": {
            "R05_teacher_minus_R04_direct": {},
            "R12_logits_minus_feature": {},
        },
    })
    wave1 = {
        "status": "PASS",
        "runs": [{"experiment_id": experiment_id} for experiment_id in sorted(closure.HISTORICAL_WAVE1)],
        "v1_test_accessed": False,
        "protected_external_surface_accessed": False,
    }
    wave2 = {
        "status": "PASS",
        "runs": [{"experiment_id": experiment_id} for experiment_id in sorted(closure.HISTORICAL_WAVE2_REQUIRED)],
        "v1_test_accessed": False,
        "protected_external_surface_accessed": False,
    }
    return direct, auxiliary, wave1, wave2


class TrackAV12ComprehensiveClosureTests(unittest.TestCase):
    def build(self, selection_status="SELECTED"):
        direct, auxiliary, wave1, wave2 = fixtures(selection_status)
        return closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={"a": "1" * 64},
            expected_closure_source_git_commit=SOURCE,
        )

    def test_selected_primary_closes_21_states_and_opens_handoff(self):
        result = self.build("SELECTED")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["track_a_comprehensive_closed"])
        self.assertEqual(result["scientific_state_count"], 21)
        self.assertEqual(len(result["scientific_state_inventory"]), 21)
        self.assertTrue(result["track_b_handoff_authorized"])
        self.assertTrue(result["track_c_handoff_authorized"])
        self.assertFalse(result["deployment_tie_gate_required"])
        self.assertEqual(result["closure_source_git_commit"], SOURCE)
        self.assertEqual(result["analysis_source_git_commit"], SOURCE)

    def test_co_primary_tie_closes_track_a_but_keeps_handoffs_closed(self):
        result = self.build("CO_PRIMARY_TIE")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["track_a_comprehensive_closed"])
        self.assertTrue(result["deployment_tie_gate_required"])
        self.assertFalse(result["track_b_handoff_authorized"])
        self.assertFalse(result["track_c_handoff_authorized"])

    def test_missing_r12_state_fails_comprehensive_closure(self):
        direct, auxiliary, wave1, wave2 = fixtures()
        auxiliary.pop("closure_sha256")
        auxiliary["state_inventory"] = auxiliary["state_inventory"][:-1]
        auxiliary = seal(auxiliary)
        result = closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={},
            expected_closure_source_git_commit=SOURCE,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["track_b_handoff_authorized"])

    def test_direct_selector_cannot_bypass_comprehensive_gate(self):
        direct, auxiliary, wave1, wave2 = fixtures()
        direct.pop("closure_sha256")
        direct["track_b_handoff_authorized"] = True
        direct = seal(direct)
        result = closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={},
            expected_closure_source_git_commit=SOURCE,
        )
        self.assertEqual(result["status"], "FAIL")

    def test_tampered_upstream_closure_hash_fails(self):
        direct, auxiliary, wave1, wave2 = fixtures()
        direct["closure_sha256"] = "0" * 64
        result = closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={},
            expected_closure_source_git_commit=SOURCE,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("direct-selection self-hash mismatch", result["errors"])

    def test_mismatched_closure_or_analysis_source_fails(self):
        direct, auxiliary, wave1, wave2 = fixtures()
        result = closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={},
            expected_closure_source_git_commit="b" * 40,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("direct-selection closure source mismatch", result["errors"])
        self.assertIn("auxiliary-analysis closure source mismatch", result["errors"])
        self.assertIn("direct-selection analysis source mismatch", result["errors"])
        self.assertIn("auxiliary-analysis analysis source mismatch", result["errors"])

    def test_wave2_superset_is_rejected(self):
        direct, auxiliary, wave1, wave2 = fixtures()
        wave2["runs"].append({"experiment_id": "R99-UNAUTHORIZED-S1"})
        result = closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={},
            expected_closure_source_git_commit=SOURCE,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("historical Wave-2 R06/R07/R12-S1 closure mismatch", result["errors"])

    def test_real_historical_closure_artifacts_match_integration_contract(self):
        evidence = ROOT / "journal_extension" / "evidence" / "public" / "track_a"
        wave1 = json.loads((evidence / "WAVE1_PRINCIPAL_VALIDATION_CLOSURE.json").read_text(encoding="utf-8"))
        wave2 = json.loads((evidence / "WAVE2_SECONDARY_VALIDATION_CLOSURE.json").read_text(encoding="utf-8"))
        direct, auxiliary, _synthetic_wave1, _synthetic_wave2 = fixtures()
        result = closure.build_comprehensive_closure(
            direct_selection=direct,
            auxiliary_analysis=auxiliary,
            wave1=wave1,
            wave2=wave2,
            evidence_hashes={"historical": "1" * 64},
            expected_closure_source_git_commit=SOURCE,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["historical_wave_closures_verified"])
        self.assertEqual(result["scientific_state_count"], 21)


if __name__ == "__main__":
    unittest.main()
