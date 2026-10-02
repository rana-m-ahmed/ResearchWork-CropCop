from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json

WAVE1 = "journal_extension/evidence/public/track_a/WAVE1_PRINCIPAL_VALIDATION_CLOSURE.json"
WAVE2 = "journal_extension/evidence/public/track_a/WAVE2_SECONDARY_VALIDATION_CLOSURE.json"


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def run(command: list[str], *, cwd: Path) -> None:
    cp = subprocess.run(command, cwd=cwd, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"closure subprocess failed rc={cp.returncode}: {' '.join(command)}")


def verify_closure_hash(payload: dict) -> bool:
    observed = payload.get("closure_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    clean = dict(payload)
    clean.pop("closure_sha256", None)
    return observed == sha256_json(clean)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--global-evidence-audit", required=True)
    ap.add_argument("--direct-evidence-index", required=True)
    ap.add_argument("--auxiliary-evidence-index", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    analysis_sha = args.analysis_source_git_commit
    if git_head(repo) != analysis_sha:
        raise SystemExit("Track-A closure controller requires the exact frozen analysis checkout")

    audit_path = Path(args.global_evidence_audit).resolve()
    direct_index = Path(args.direct_evidence_index).resolve()
    auxiliary_index = Path(args.auxiliary_evidence_index).resolve()
    audit = load_json(audit_path)
    if (
        audit.get("status") != "PASS"
        or audit.get("audit_kind") != "track_a_posttraining_global_evidence"
        or audit.get("analysis_source_git_commit") != analysis_sha
        or audit.get("scientific_state_count") != 21
        or audit.get("direct_state_count") != 12
        or audit.get("private_roundtrip_verified_state_count") != 21
        or audit.get("selector_authorized") is not True
        or audit.get("auxiliary_analysis_authorized") is not True
        or audit.get("v1_test_accessed") is not False
        or audit.get("external_predictions_opened") is not False
        or audit.get("track_c_candidate_results_opened") is not False
    ):
        raise SystemExit("global evidence audit does not authorize Track-A closure")
    if sha256_file(direct_index) != audit.get("direct_evidence_index_sha256"):
        raise SystemExit("direct evidence index differs from independently audited index")
    if sha256_file(auxiliary_index) != audit.get("auxiliary_evidence_index_sha256"):
        raise SystemExit("auxiliary evidence index differs from independently audited index")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise SystemExit("Track-A closure output directory must not already exist")
    output.mkdir(parents=True, exist_ok=False)

    auxiliary_analysis = output / "TRACKA_AUXILIARY_ANALYSIS.json"
    direct_selection = output / "TRACKA_DIRECT_SELECTION.json"
    comprehensive = output / "TRACKA_COMPREHENSIVE_CLOSURE.json"

    run([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/seal_tracka_v12_auxiliary_analysis.py"),
        "--repo-root", str(repo),
        "--evidence-index", str(auxiliary_index),
        "--output", str(auxiliary_analysis),
    ], cwd=repo)
    aux = load_json(auxiliary_analysis)
    if aux.get("status") != "PASS" or not verify_closure_hash(aux):
        raise SystemExit("auxiliary analysis did not seal a valid PASS closure")

    run([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/seal_tracka_v12_selection.py"),
        "--repo-root", str(repo),
        "--evidence-index", str(direct_index),
        "--output", str(direct_selection),
    ], cwd=repo)
    selection = load_json(direct_selection)
    if (
        selection.get("status") != "PASS"
        or selection.get("science_selection_sealed") is not True
        or selection.get("track_b_handoff_authorized") is not False
        or selection.get("track_c_handoff_authorized") is not False
        or not verify_closure_hash(selection)
    ):
        raise SystemExit("direct selector did not produce the required sealed pre-handoff PASS")

    wave1 = repo / WAVE1
    wave2 = repo / WAVE2
    run([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/seal_tracka_v12_comprehensive_closure.py"),
        "--repo-root", str(repo),
        "--direct-selection", str(direct_selection),
        "--auxiliary-analysis", str(auxiliary_analysis),
        "--wave1-closure", str(wave1),
        "--wave2-closure", str(wave2),
        "--output", str(comprehensive),
    ], cwd=repo)
    closure = load_json(comprehensive)
    if (
        closure.get("status") != "PASS"
        or closure.get("track_a_comprehensive_closed") is not True
        or closure.get("scientific_state_count") != 21
        or closure.get("analysis_source_git_commit") != analysis_sha
        or closure.get("v1_test_accessed") is not False
        or closure.get("external_predictions_opened_before_selection") is not False
        or closure.get("track_c_candidate_results_opened_before_selection") is not False
        or not verify_closure_hash(closure)
    ):
        raise SystemExit("comprehensive Track-A closure failed independent terminal checks")

    selection_status = (closure.get("selection") or {}).get("status")
    tie = selection_status == "CO_PRIMARY_TIE"
    if selection_status not in {"SELECTED", "CO_PRIMARY_TIE"}:
        raise SystemExit("comprehensive closure selection terminal status invalid")
    if tie:
        if closure.get("deployment_tie_gate_required") is not True:
            raise SystemExit("co-primary tie did not require deployment feasibility gate")
        if closure.get("track_b_handoff_authorized") is not False or closure.get("track_c_handoff_authorized") is not False:
            raise SystemExit("co-primary tie improperly opened downstream handoffs")
    else:
        if closure.get("deployment_tie_gate_required") is not False:
            raise SystemExit("single selected primary unexpectedly requires tie gate")
        if closure.get("track_b_handoff_authorized") is not True or closure.get("track_c_handoff_authorized") is not True:
            raise SystemExit("comprehensive SELECTED closure failed to authorize sealed-primary handoff")

    final_audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "audit_kind": "track_a_final_closure",
        "analysis_source_git_commit": analysis_sha,
        "global_evidence_audit_sha256": sha256_file(audit_path),
        "direct_evidence_index_sha256": sha256_file(direct_index),
        "auxiliary_evidence_index_sha256": sha256_file(auxiliary_index),
        "auxiliary_analysis_sha256": sha256_file(auxiliary_analysis),
        "direct_selection_sha256": sha256_file(direct_selection),
        "comprehensive_closure_sha256": sha256_file(comprehensive),
        "selection_terminal_status": selection_status,
        "track_a_closed": True,
        "scientific_state_count": 21,
        "direct_candidate_state_count": 12,
        "auxiliary_state_count": 9,
        "private_evidence_roundtrip_verified_state_count": 21,
        "direct_selection_was_downstream_blind": True,
        "v1_test_accessed": False,
        "external_predictions_opened_before_selection": False,
        "track_c_candidate_results_opened_before_selection": False,
        "training_reperformed": False,
        "optimizer_state_advanced": False,
        "downstream_handoff_authorized": not tie,
        "deployment_tie_gate_required": tie,
    }
    final_audit["final_audit_sha256"] = sha256_json(final_audit)
    final_path = output / "TRACKA_FINAL_CLOSURE_AUDIT.json"
    atomic_write_json(final_path, final_audit)
    print(json.dumps(final_audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
