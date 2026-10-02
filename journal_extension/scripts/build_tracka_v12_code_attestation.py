from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json

STATIC_GATES = (
    "exact_head_ci",
    "config_contract_and_failure_injection",
    "principal_science_diff",
    "secondary_science_diff",
    "training_runner_contract",
    "teacher_factory_root_binding_contract",
    "checkpoint_recovery_contract",
    "cross_slot_recovery_contract",
    "kaggle_generation_durability_contract",
    "g1a_runtime_global_resolution_contract",
    "r13_parity_amendment_contract",
    "six_gpu_parent_orchestration",
    "selected_checkpoint_replay_implementation",
    "classwise_confusion_implementation",
    "robustness_executor_implementation",
    "efficiency_profiler_implementation",
    "xai_gradcampp_implementation",
    "analysis_selector_implementation",
    "direct_selection_closure_implementation",
    "auxiliary_replay_implementation",
    "auxiliary_paired_analysis_implementation",
    "comprehensive_21_state_closure_implementation",
)

IMPLEMENTATION_FILES = (
    "journal_extension/TRACK_A_V12_EXECUTION_RUNBOOK.md",
    "journal_extension/src/cropcop_je/checkpointing.py",
    "journal_extension/src/cropcop_je/persistence.py",
    "journal_extension/src/cropcop_je/persistence_v8.py",
    "journal_extension/src/cropcop_je/secondary.py",
    "journal_extension/src/cropcop_je/tracka_v12.py",
    "journal_extension/src/cropcop_je/tracka_v12_analysis.py",
    "journal_extension/src/cropcop_je/tracka_v12_authorization.py",
    "journal_extension/src/cropcop_je/tracka_v12_evidence.py",
    "journal_extension/src/cropcop_je/tracka_v12_g1a.py",
    "journal_extension/src/cropcop_je/tracka_v12_g1a_v121.py",
    "journal_extension/src/cropcop_je/tracka_v12_g2a.py",
    "journal_extension/src/cropcop_je/tracka_v12_g2a_durability.py",
    "journal_extension/src/cropcop_je/tracka_v12_g2a_v122.py",
    "journal_extension/src/cropcop_je/tracka_v12_historical.py",
    "journal_extension/src/cropcop_je/tracka_v12_orchestration.py",
    "journal_extension/src/cropcop_je/tracka_v12_placement.py",
    "journal_extension/src/cropcop_je/tracka_v12_posttraining.py",
    "journal_extension/src/cropcop_je/tracka_v12_runtime.py",
    "journal_extension/src/cropcop_je/tracka_v12_xai.py",
    "journal_extension/teacher_factory/historical_dino_tiny.py",
    "journal_extension/scripts/seal_tracka_v12_g1a.py",
    "journal_extension/scripts/seal_tracka_v12_g1a_v121.py",
    "journal_extension/scripts/qualify_tracka_v12_profile_v122.py",
    "journal_extension/scripts/seal_tracka_v12_g2a.py",
    "journal_extension/scripts/run_tracka_v12_training_v121.py",
    "journal_extension/scripts/run_tracka_v12_direct_evidence.py",
    "journal_extension/scripts/run_tracka_v12_auxiliary_evidence.py",
    "journal_extension/scripts/run_tracka_v12_xai.py",
    "journal_extension/scripts/seal_tracka_v12_selection.py",
    "journal_extension/scripts/seal_tracka_v12_auxiliary_analysis.py",
    "journal_extension/scripts/seal_tracka_v12_comprehensive_closure.py",
    "journal_extension/scripts/build_tracka_v12_code_attestation.py",
    "journal_extension/scripts/build_tracka_v12_lock_runtime_attestation.py",
    "journal_extension/scripts/seal_tracka_v12_science_go.py",
    "journal_extension/scripts/seal_tracka_v12_science_go_v123.py",
    "journal_extension/scripts/seal_tracka_v12_science_go_v124.py",
    "journal_extension/kaggle/run_tracka_v12_account.py",
    "journal_extension/kaggle/run_tracka_v12_account_v121.py",
    "journal_extension/amendments/track_a_strengthening_v1/model_selection_operationalization_v1_2_1.json",
    "journal_extension/amendments/track_a_strengthening_v1/parallel_execution_plan_v1_2_2.json",
    "journal_extension/amendments/track_a_strengthening_v1/candidate_comparison_claim_boundary_v1_2_1.json",
    "journal_extension/amendments/track_a_strengthening_v1/xai_operationalization_v1_2_2.json",
    "journal_extension/amendments/track_a_strengthening_v1/R13_PARITY_PREEXECUTION_AMENDMENT_v1_2_1.md",
    "journal_extension/amendments/track_a_strengthening_v1/r13_parity_preexecution_evidence_v1_2_1.json",
    "journal_extension/amendments/track_a_strengthening_v1/r13_pretrained_identity_and_normalization_contract_v1_2_1.json",
    "journal_extension/amendments/track_a_strengthening_v1/AMENDMENT_V1_2_1_PARITY_CONTENT_LOCK.json",
    "journal_extension/amendments/track_a_strengthening_v1/validate_r13_parity_amendment_v1_2_1.py",
    "journal_extension/amendments/track_a_strengthening_v1/validate_r13_pretrained_parity_v1_2_1.py",
    "tests/test_tracka_v12_teacher_factory_root_binding.py",
)


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--expected-source-sha", required=True)
    ap.add_argument("--principal-science-diff", required=True)
    ap.add_argument("--secondary-science-diff", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("code attestation may only be emitted by GitHub Actions after the gated test sequence")
    repo = Path(args.repo_root).resolve()
    head = git_head(repo)
    expected = args.expected_source_sha.strip()
    if len(expected) != 40 or head != expected:
        raise SystemExit(f"exact-head mismatch: expected={expected}, checkout={head}")

    principal = Path(args.principal_science_diff).resolve()
    secondary = Path(args.secondary_science_diff).resolve()
    if not principal.is_file() or not secondary.is_file():
        raise SystemExit("science-diff reports are missing")
    for path in IMPLEMENTATION_FILES:
        if not (repo / path).is_file():
            raise SystemExit(f"required implementation file missing: {path}")

    payload = {
        "schema_version": "1.1",
        "attestation_kind": "track_a_v12_pre_science_code",
        "status": "PASS",
        "source_git_commit": head,
        "github_actions": True,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "pull_request_head_sha": expected,
        "static_pre_science_gates": {name: "PASS" for name in STATIC_GATES},
        "implementation_sha256": {path: sha256_file(repo / path) for path in IMPLEMENTATION_FILES},
        "science_diff_sha256": {
            "principal": sha256_file(principal),
            "secondary": sha256_file(secondary),
        },
        "science_diff_reports": {
            "principal": json.loads(principal.read_text(encoding="utf-8")),
            "secondary": json.loads(secondary.read_text(encoding="utf-8")),
        },
        "r13_parity_contract_id": "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1",
        "r13_parity_required_max_abs_difference": 5e-5,
        "note": "This attestation is emitted only after exact-head contract tests, failure injection, orchestration/placement tests, G1A runtime-global checks, generation-aware Kaggle durability tests, the historical-teacher factory source-root binding regression, versioned R13 parity-amendment validation, versioned G1A consumer-binding tests, historical-lineage tests, analysis-evidence tests and both historical science-diff sentinels pass in GitHub Actions. It binds the canonical runbook plus G1A/G2A qualification, the exact historical teacher factory source bytes, parent account preflight, durability-bound SCIENCE_GO construction, checkpoint/persistence recovery, six-GPU execution, historical replay, direct/auxiliary analysis, XAI, direct selection and comprehensive 21-state closure implementations. The separate exact-head lock/runtime attestation additionally binds the exact-pretrained R13 parity proof. Neither attestation independently authorizes scientific execution.",
    }
    payload["attestation_sha256"] = sha256_json(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
