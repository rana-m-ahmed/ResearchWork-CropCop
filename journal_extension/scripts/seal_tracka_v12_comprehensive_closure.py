from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_evidence import ALL_DIRECT_STATES

SEEDS = ("S1", "S2", "S3")
R04 = {f"R04-MNV4-DIRECT-{seed}" for seed in SEEDS}
R05 = {f"R05-MNV4-TEACHER-{seed}" for seed in SEEDS}
R12 = {f"R12-MNV4-{mode}-{seed}" for mode in ("LOGITS", "FEATURE") for seed in SEEDS}
AUXILIARY_ONLY = R05 | R12
FULL_TRACK_A = set(ALL_DIRECT_STATES) | AUXILIARY_ONLY
HISTORICAL_WAVE1 = R04 | R05
HISTORICAL_WAVE2_REQUIRED = {
    "R12-MNV4-LOGITS-S1",
    "R12-MNV4-FEATURE-S1",
    "R06-EFFB0-CONTEXT-S1",
    "R07-CNXTT-CONTEXT-S1",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_comprehensive_closure(
    *,
    direct_selection: dict[str, Any],
    auxiliary_analysis: dict[str, Any],
    wave1: dict[str, Any],
    wave2: dict[str, Any],
    evidence_hashes: dict[str, str],
) -> dict[str, Any]:
    errors: list[str] = []
    if direct_selection.get("status") != "PASS" or direct_selection.get("science_selection_sealed") is not True:
        errors.append("direct model selection is not sealed PASS")
    if direct_selection.get("closure_kind") != "track_a_direct_model_selection":
        errors.append("unexpected direct-selection closure kind")
    if set(direct_selection.get("direct_state_inventory", [])) != set(ALL_DIRECT_STATES):
        errors.append("direct-selection state inventory mismatch")
    if direct_selection.get("track_b_handoff_authorized") is not False or direct_selection.get("track_c_handoff_authorized") is not False:
        errors.append("direct-selection artifact bypassed comprehensive handoff gate")
    if direct_selection.get("v1_test_accessed") is not False or direct_selection.get("external_predictions_opened") is not False:
        errors.append("direct-selection protected surface marker invalid")

    if auxiliary_analysis.get("status") != "PASS" or auxiliary_analysis.get("closure_kind") != "track_a_auxiliary_three_seed_analysis":
        errors.append("auxiliary three-seed analysis is not PASS")
    auxiliary_inventory = set(auxiliary_analysis.get("state_inventory", []))
    expected_aux_analysis_inventory = R04 | R05 | R12
    if auxiliary_inventory != expected_aux_analysis_inventory:
        errors.append("auxiliary-analysis state inventory mismatch")
    if auxiliary_analysis.get("v1_test_accessed") is not False or auxiliary_analysis.get("external_surface_accessed") is not False:
        errors.append("auxiliary-analysis protected surface marker invalid")
    paired = auxiliary_analysis.get("paired_analyses", {})
    if set(paired) != {"R05_teacher_minus_R04_direct", "R12_logits_minus_feature"}:
        errors.append("auxiliary paired-analysis inventory mismatch")
    if auxiliary_analysis.get("hypothesis_tests_authorized") is not False:
        errors.append("auxiliary analysis unexpectedly authorizes hypothesis tests")

    wave1_runs = {row.get("experiment_id") for row in wave1.get("runs", [])}
    if wave1.get("status") != "PASS" or wave1_runs != HISTORICAL_WAVE1:
        errors.append("historical Wave-1 R04/R05 closure mismatch")
    if wave1.get("v1_test_accessed") is not False or wave1.get("protected_external_surface_accessed") is not False:
        errors.append("historical Wave-1 protected surface marker invalid")

    wave2_runs = {row.get("experiment_id") for row in wave2.get("runs", [])}
    if wave2.get("status") != "PASS" or not HISTORICAL_WAVE2_REQUIRED.issubset(wave2_runs):
        errors.append("historical Wave-2 R06/R07/R12-S1 closure mismatch")
    if wave2.get("v1_test_accessed") is not False or wave2.get("protected_external_surface_accessed") is not False:
        errors.append("historical Wave-2 protected surface marker invalid")

    union = set(ALL_DIRECT_STATES) | (auxiliary_inventory - R04)
    if union != FULL_TRACK_A or len(union) != 21:
        errors.append("comprehensive Track-A inventory does not resolve to exactly 21 unique scientific states")

    selection = direct_selection.get("selection", {})
    selection_status = selection.get("status")
    if selection_status not in {"SELECTED", "CO_PRIMARY_TIE"}:
        errors.append("direct selection result has unsupported terminal status")

    if errors:
        return {
            "schema_version": "1.0",
            "status": "FAIL",
            "closure_kind": "track_a_comprehensive_21_state_closure",
            "track_a_comprehensive_closed": False,
            "track_b_handoff_authorized": False,
            "track_c_handoff_authorized": False,
            "errors": errors,
        }

    tie = selection_status == "CO_PRIMARY_TIE"
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "closure_kind": "track_a_comprehensive_21_state_closure",
        "track_a_comprehensive_closed": True,
        "scientific_state_count": 21,
        "scientific_state_inventory": sorted(FULL_TRACK_A),
        "direct_candidate_state_count": 12,
        "teacher_comparator_state_count": 3,
        "mechanism_state_count": 6,
        "selection": selection,
        "deployment_tie_gate_required": tie,
        "track_b_handoff_authorized": not tie,
        "track_c_handoff_authorized": not tie,
        "v1_test_accessed": False,
        "external_predictions_opened_before_selection": False,
        "track_c_candidate_results_opened_before_selection": False,
        "historical_wave_closures_verified": True,
        "auxiliary_hypothesis_tests_authorized": False,
        "evidence_sha256": evidence_hashes,
        "claim_boundary": "Track A selects the best frozen pretrained candidate system under the common CropCop downstream protocol; it does not identify a causal architecture-only effect.",
        "note": (
            "Track A is one comprehensive 21-state journal experiment executed in computational waves. "
            "If the direct selector returns SELECTED, Track B and Track C may now receive only that sealed scientific-primary family. "
            "If it returns CO_PRIMARY_TIE, both handoffs remain closed until the separately frozen deployment-feasibility tie gate resolves the tie without another Track-A experiment."
        ),
        "errors": [],
    }
    result["closure_sha256"] = sha256_json(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct-selection", required=True)
    ap.add_argument("--auxiliary-analysis", required=True)
    ap.add_argument("--wave1-closure", required=True)
    ap.add_argument("--wave2-closure", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    paths = {
        "direct_selection": Path(args.direct_selection).resolve(),
        "auxiliary_analysis": Path(args.auxiliary_analysis).resolve(),
        "wave1": Path(args.wave1_closure).resolve(),
        "wave2": Path(args.wave2_closure).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise SystemExit("comprehensive Track-A closure input missing")
    payloads = {name: load_json(path) for name, path in paths.items()}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    result = build_comprehensive_closure(
        direct_selection=payloads["direct_selection"],
        auxiliary_analysis=payloads["auxiliary_analysis"],
        wave1=payloads["wave1"],
        wave2=payloads["wave2"],
        evidence_hashes=hashes,
    )
    if result.get("status") != "PASS":
        raise SystemExit("comprehensive Track-A closure failed: " + json.dumps(result, sort_keys=True))
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
