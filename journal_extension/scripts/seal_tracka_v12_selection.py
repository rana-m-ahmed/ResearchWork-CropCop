from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_analysis import FAMILIES, ROBUSTNESS_CORRUPTIONS, ROBUSTNESS_SEVERITIES, SEED_LABELS
from cropcop_je.tracka_v12_evidence import (
    ALL_DIRECT_STATES,
    DIRECT_STATES,
    build_family_selector_row,
    build_tracka_selection_closure,
    validate_direct_state_evidence_bundle,
)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def seed_for_state(experiment_id: str) -> str:
    seed = experiment_id.rsplit("-", 1)[-1]
    if seed not in SEED_LABELS:
        raise ValueError(f"cannot resolve frozen seed label from {experiment_id}")
    return seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--evidence-index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    closure_source_git_commit = git_head(repo)
    if len(closure_source_git_commit) != 40:
        raise SystemExit("direct-selection checkout is not bound to a full Git SHA")

    index_path = Path(args.evidence_index).resolve()
    index = load_json(index_path)
    states = index.get("states", {})
    if index.get("schema_version") != "1.0" or set(states) != set(ALL_DIRECT_STATES):
        raise SystemExit("final selection evidence index must contain the exact 12 direct candidate states")

    state_gates = {}
    xai_gates = {}
    evidence_hashes = {}
    per_family_seed_summary = {family: {} for family in FAMILIES}
    per_family_corruption = {family: {} for family in FAMILIES}
    per_family_efficiency = {family: {} for family in FAMILIES}
    xai_targets = {family: {} for family in FAMILIES}
    xai_warnings = []

    family_for_state = {
        experiment_id: family
        for family, experiment_ids in DIRECT_STATES.items()
        for experiment_id in experiment_ids
    }

    for experiment_id in ALL_DIRECT_STATES:
        row = states[experiment_id]
        required_keys = {"direct_gate", "robustness_replay", "efficiency", "xai_gate"}
        if set(row) != required_keys:
            raise SystemExit(f"evidence index field mismatch for {experiment_id}")
        paths = {name: Path(value).resolve() for name, value in row.items()}
        if any(not path.is_file() for path in paths.values()):
            raise SystemExit(f"evidence file missing for {experiment_id}")
        direct = load_json(paths["direct_gate"])
        robust = load_json(paths["robustness_replay"])
        efficiency = load_json(paths["efficiency"])
        xai = load_json(paths["xai_gate"])

        bundle_errors = validate_direct_state_evidence_bundle(
            experiment_id,
            direct=direct,
            robustness=robust,
            efficiency=efficiency,
            xai=xai,
        )
        if bundle_errors:
            raise SystemExit(f"direct evidence bundle failed for {experiment_id}: " + "; ".join(bundle_errors))

        actual_efficiency_sha = sha256_file(paths["efficiency"])
        actual_robustness_sha = sha256_file(paths["robustness_replay"])
        bound_public = direct.get("public_evidence_sha256", {})
        if bound_public.get("efficiency.json") != actual_efficiency_sha:
            raise SystemExit(f"direct gate/efficiency file hash mismatch for {experiment_id}")
        if bound_public.get("robustness_and_replay.json") != actual_robustness_sha:
            raise SystemExit(f"direct gate/robustness file hash mismatch for {experiment_id}")

        direct_ok = direct.get("status") == "PASS"
        replay_ok = direct.get("replay_gate", {}).get("status") == "PASS"
        state_gates[experiment_id] = {
            "terminal": direct_ok,
            "selected_checkpoint_verified": bool(direct.get("selected_checkpoint_sha256")),
            "replay_pass": replay_ok,
            "robustness_pass": direct.get("robustness_pass") is True,
            "classwise_pass": direct.get("classwise_pass") is True,
            "efficiency_pass": direct.get("efficiency_pass") is True,
            "v1_test_accessed": direct.get("v1_test_accessed"),
            "external_surface_accessed": direct.get("external_surface_accessed"),
        }

        xai_status = str(xai.get("status", ""))
        if xai_status != "PASS":
            xai_warnings.append({
                "experiment_id": experiment_id,
                "status": xai_status,
                "finite_map_rate": xai.get("finite_map_rate"),
                "degenerate_map_rate": xai.get("degenerate_map_rate"),
            })
        xai_gates[experiment_id] = {
            "status": "PASS",
            "reported_status": xai_status,
            "training_or_adaptation_performed": xai.get("training_or_adaptation_performed"),
        }

        family = family_for_state[experiment_id]
        seed = seed_for_state(experiment_id)
        clean = robust["clean_summary"]
        per_family_seed_summary[family][seed] = clean
        cells = robust["cells"]
        per_family_corruption[family][seed] = {
            corruption: {
                severity: float(cells[corruption][severity]["validation_macro_f1"])
                for severity in ROBUSTNESS_SEVERITIES
            }
            for corruption in ROBUSTNESS_CORRUPTIONS
        }
        per_family_efficiency[family][seed] = {
            "model_state_tensor_bytes_fp32": int(efficiency["model_state_tensor_bytes_fp32"]),
            "total_parameter_count": int(efficiency["total_parameter_count"]),
            "trainable_parameter_count": int(efficiency["trainable_parameter_count"]),
            "input_resolution": int(efficiency["input_resolution"]),
        }
        xai_targets[family][seed] = str(xai["target_module_path"])
        evidence_hashes[experiment_id] = {
            "direct_gate": sha256_file(paths["direct_gate"]),
            "robustness_replay": actual_robustness_sha,
            "efficiency": actual_efficiency_sha,
            "xai_gate": sha256_file(paths["xai_gate"]),
            "selected_checkpoint_sha256": direct["selected_checkpoint_sha256"],
            "scientific_source_git_commit": direct["source_git_commit"],
        }

    for family in FAMILIES:
        if len(set(xai_targets[family].values())) != 1:
            raise SystemExit(f"XAI target path differs across seeds for {family}")

    selector_rows = [
        build_family_selector_row(
            family,
            seed_summaries=per_family_seed_summary[family],
            corruption_macro_f1=per_family_corruption[family],
            efficiency_by_seed=per_family_efficiency[family],
        )
        for family in FAMILIES
    ]
    closure = build_tracka_selection_closure(
        selector_rows=selector_rows,
        state_gates=state_gates,
        xai_gates=xai_gates,
    )
    if closure.get("status") != "PASS":
        raise SystemExit("Track-A direct model-selection closure failed: " + json.dumps(closure, sort_keys=True))

    result = {
        **closure,
        "closure_kind": "track_a_direct_model_selection",
        "closure_source_git_commit": closure_source_git_commit,
        "evidence_index_sha256": sha256_file(index_path),
        "evidence_sha256": evidence_hashes,
        "selector_rows": [asdict(row) for row in selector_rows],
        "xai_target_paths_by_family_seed": xai_targets,
        "xai_methodological_warnings": xai_warnings,
        "state_gates": state_gates,
        "track_a_comprehensive_closed": False,
        "track_b_handoff_authorized": False,
        "track_c_handoff_authorized": False,
        "note": "This artifact seals the scientific-primary selection among the 12 direct candidate states only. Track B/C remain closed until the separate comprehensive Track-A closure also verifies the R05 teacher-effect comparator and all six R12 mechanism states. Neither Track-B nor Track-C evidence participated in this selection.",
    }
    result["closure_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
