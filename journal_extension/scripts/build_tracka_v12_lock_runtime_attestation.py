from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json


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
    if not content_path.is_file() or not runtime_path.is_file():
        raise SystemExit("content-lock/runtime report missing")
    content = load_json(content_path)
    runtime = load_json(runtime_path)
    if content.get("overall_status") != "PASS" or content.get("static", {}).get("status") != "PASS":
        raise SystemExit("immutable v1.2 content-lock report is not PASS")
    if content.get("static", {}).get("science_authorized") is not False:
        raise SystemExit("content-lock report unexpectedly authorizes science")
    if runtime.get("status") != "PASS" or runtime.get("science_authorized") is not False:
        raise SystemExit("R13 v1.2.1 runtime qualification is not pre-science PASS")
    if runtime.get("qualified_target") != "blocks.13.norm1":
        raise SystemExit("R13 runtime qualification target drift")
    if runtime.get("observed_target_shape") != [2, 257, 320]:
        raise SystemExit("R13 runtime qualification tensor contract drift")

    claim = repo / "journal_extension/amendments/track_a_strengthening_v1/candidate_comparison_claim_boundary_v1_2_1.json"
    xai = repo / "journal_extension/amendments/track_a_strengthening_v1/xai_operationalization_v1_2_2.json"
    if not claim.is_file() or not xai.is_file():
        raise SystemExit("pre-science claim/XAI operationalization lock missing")

    payload = {
        "schema_version": "1.0",
        "attestation_kind": "track_a_v12_exact_head_lock_runtime",
        "status": "PASS",
        "source_git_commit": head,
        "github_actions": True,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "pull_request_head_sha": expected,
        "immutable_v12_lock": "PASS",
        "v121_runtime_qualification": "PASS",
        "candidate_claim_boundary_lock": "PASS",
        "content_lock_report_sha256": sha256_file(content_path),
        "runtime_report_sha256": sha256_file(runtime_path),
        "candidate_claim_boundary_sha256": sha256_file(claim),
        "xai_operationalization_sha256": sha256_file(xai),
        "content_lock_report": content,
        "runtime_report": runtime,
        "science_authorized": False,
        "note": "This exact-head attestation binds the immutable v1.2 content lock, corrected R13 v1.2.1 runtime/XAI interface qualification, candidate-system claim boundary, and frozen XAI operationalization to one Git commit. It does not independently authorize scientific execution.",
    }
    payload["attestation_sha256"] = sha256_json(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
