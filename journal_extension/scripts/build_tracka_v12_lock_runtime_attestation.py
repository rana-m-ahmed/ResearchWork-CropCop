from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json

PARITY_CONTRACT_ID = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1"
PARITY_TOLERANCE = 5e-5
R13_PRETRAINED_SHA256 = "92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--expected-source-sha", required=True)
    ap.add_argument("--content-lock-report", required=True)
    ap.add_argument("--runtime-report", required=True)
    ap.add_argument("--parity-amendment-report", required=True)
    ap.add_argument("--exact-pretrained-parity-report", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("lock/runtime attestation may only be emitted by GitHub Actions")
    repo = Path(args.repo_root).resolve()
    head = git_head(repo)
    expected = args.expected_source_sha.strip()
    if len(expected) != 40 or head != expected:
        raise SystemExit(f"exact-head mismatch: expected={expected}, checkout={head}")

    content_path = Path(args.content_lock_report).resolve()
    runtime_path = Path(args.runtime_report).resolve()
    parity_path = Path(args.parity_amendment_report).resolve()
    exact_path = Path(args.exact_pretrained_parity_report).resolve()
    for path in (content_path, runtime_path, parity_path, exact_path):
        if not path.is_file():
            raise SystemExit(f"lock/runtime attestation input missing: {path}")

    content = load_json(content_path)
    runtime = load_json(runtime_path)
    parity = load_json(parity_path)
    exact = load_json(exact_path)

    if content.get("overall_status") != "PASS" or content.get("static", {}).get("status") != "PASS":
        raise SystemExit("immutable v1.2 content-lock report is not PASS")
    if content.get("static", {}).get("science_authorized") is not False:
        raise SystemExit("content-lock report unexpectedly authorizes science")
    if runtime.get("status") != "PASS" or runtime.get("science_authorized") is not False:
        raise SystemExit("R13 v1.2.1 architecture/runtime qualification is not pre-science PASS")
    if runtime.get("qualified_target") != "blocks.13.norm1":
        raise SystemExit("R13 runtime qualification target drift")
    if runtime.get("observed_target_shape") != [2, 257, 320]:
        raise SystemExit("R13 runtime qualification tensor contract drift")

    if parity.get("status") != "PASS" or parity.get("science_authorized") is not False:
        raise SystemExit("R13 parity amendment static qualification is not pre-science PASS")
    if parity.get("new_contract_id") != PARITY_CONTRACT_ID:
        raise SystemExit("R13 parity amendment contract identity drift")
    if float(parity.get("old_required_max_abs_difference", -1)) != 1e-5:
        raise SystemExit("R13 historical parity threshold provenance drift")
    if float(parity.get("new_required_max_abs_difference", -1)) != PARITY_TOLERANCE:
        raise SystemExit("R13 superseding parity threshold drift")

    if exact.get("status") != "PASS" or exact.get("science_authorized") is not False:
        raise SystemExit("exact-pretrained R13 parity qualification is not pre-science PASS")
    if exact.get("contract_id") != PARITY_CONTRACT_ID:
        raise SystemExit("exact-pretrained parity contract identity drift")
    if exact.get("pretrained_sha256") != R13_PRETRAINED_SHA256:
        raise SystemExit("exact-pretrained parity artifact identity drift")
    if float(exact.get("required_max_abs_difference", -1)) != PARITY_TOLERANCE:
        raise SystemExit("exact-pretrained parity required threshold drift")
    if exact.get("historical_v1_2_gate_pass") is not False or exact.get("v1_2_1_gate_pass") is not True:
        raise SystemExit("exact-pretrained parity report does not establish old-fail/new-pass amendment basis")
    for key in ("scientific_metric_computed", "classifier_prediction_opened", "v1_test_accessed", "external_surface_accessed"):
        if exact.get(key) is not False:
            raise SystemExit(f"exact-pretrained parity protected marker invalid: {key}")

    claim = repo / "journal_extension/amendments/track_a_strengthening_v1/candidate_comparison_claim_boundary_v1_2_1.json"
    xai = repo / "journal_extension/amendments/track_a_strengthening_v1/xai_operationalization_v1_2_2.json"
    parity_contract = repo / "journal_extension/amendments/track_a_strengthening_v1/r13_pretrained_identity_and_normalization_contract_v1_2_1.json"
    parity_evidence = repo / "journal_extension/amendments/track_a_strengthening_v1/r13_parity_preexecution_evidence_v1_2_1.json"
    parity_lock = repo / "journal_extension/amendments/track_a_strengthening_v1/AMENDMENT_V1_2_1_PARITY_CONTENT_LOCK.json"
    for path in (claim, xai, parity_contract, parity_evidence, parity_lock):
        if not path.is_file():
            raise SystemExit(f"pre-science lock input missing: {path}")

    payload = {
        "schema_version": "1.1",
        "attestation_kind": "track_a_v12_exact_head_lock_runtime",
        "status": "PASS",
        "source_git_commit": head,
        "github_actions": True,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "pull_request_head_sha": expected,
        "immutable_v12_lock": "PASS",
        "r13_v121_runtime_qualification": "PASS",
        "r13_v121_parity_amendment": "PASS",
        "r13_exact_pretrained_parity": "PASS",
        "candidate_claim_boundary_lock": "PASS",
        "content_lock_report_sha256": sha256_file(content_path),
        "runtime_report_sha256": sha256_file(runtime_path),
        "parity_amendment_report_sha256": sha256_file(parity_path),
        "exact_pretrained_parity_report_sha256": sha256_file(exact_path),
        "candidate_claim_boundary_sha256": sha256_file(claim),
        "xai_operationalization_sha256": sha256_file(xai),
        "r13_parity_contract_sha256": sha256_file(parity_contract),
        "r13_parity_preexecution_evidence_sha256": sha256_file(parity_evidence),
        "r13_parity_content_lock_sha256": sha256_file(parity_lock),
        "r13_parity_contract_id": PARITY_CONTRACT_ID,
        "r13_parity_required_max_abs_difference": PARITY_TOLERANCE,
        "content_lock_report": content,
        "runtime_report": runtime,
        "parity_amendment_report": parity,
        "exact_pretrained_parity_report": exact,
        "science_authorized": False,
        "note": "This exact-head attestation preserves the immutable v1.2 content lock while binding the pre-science R13 parity v1.2.1 amendment, exact pinned-pretrained parity proof, corrected R13 runtime/XAI interface qualification, candidate-system claim boundary, and frozen XAI operationalization to one Git commit. The parity amendment changes only the pre-execution float32 maximum acceptance bound from 1e-5 to 5e-5; it does not independently authorize scientific execution.",
    }
    payload["attestation_sha256"] = sha256_json(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
