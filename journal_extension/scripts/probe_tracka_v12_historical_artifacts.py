from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import verify_selected
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_historical import (
    HISTORICAL_TRACKA_CLOSURE_SPECS,
    validate_historical_closure_identity,
)
from cropcop_je.train import _identity as checkpoint_identity

ACCOUNTS = {"K1", "K2", "K3"}


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def check_file(path: str | Path) -> tuple[bool, str | None]:
    p = Path(path).resolve()
    return (p.is_file(), str(p) if p.is_file() else None)


def check_dir(path: str | Path) -> tuple[bool, str | None]:
    p = Path(path).resolve()
    return (p.is_dir(), str(p) if p.is_dir() else None)


def probe_state(experiment_id: str, spec: dict) -> dict:
    result = {
        "experiment_id": experiment_id,
        "available": False,
        "errors": [],
        "verified": {},
    }
    run_path = Path(spec.get("run_record", "")).expanduser().resolve()
    checkpoint_root = Path(spec.get("checkpoint_root", "")).expanduser().resolve()
    if not run_path.is_file():
        result["errors"].append("run_record_missing")
        return result
    if not checkpoint_root.is_dir():
        result["errors"].append("checkpoint_root_missing")
        return result
    try:
        record = load_json(run_path)
        errors = validate_historical_closure_identity(record)
        if errors:
            result["errors"].extend(f"historical_identity:{x}" for x in errors)
            return result
        if record.get("experiment_id") != experiment_id:
            result["errors"].append("experiment_id_mismatch")
            return result
        expected = ((record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
        if not expected:
            expected = (record.get("result_summary") or {}).get("selected_checkpoint_sha256")
        checkpoint_path, payload = verify_selected(
            checkpoint_root,
            expected_identity=checkpoint_identity(record),
            expected_sha256=str(expected),
        )
        for path in spec.get("required_executor_files", []):
            ok, _ = check_file(path)
            if not ok:
                result["errors"].append(f"required_file_missing:{path}")
        for path in spec.get("required_executor_directories", []):
            ok, _ = check_dir(path)
            if not ok:
                result["errors"].append(f"required_directory_missing:{path}")
        result["verified"] = {
            "run_id": record.get("run_id"),
            "run_record_sha256": sha256_file(run_path),
            "selected_checkpoint_sha256": str(expected),
            "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
            "selected_checkpoint_epoch": int(payload["epoch"]),
            "scientific_source_git_commit": record.get("source_git_commit"),
        }
        result["available"] = not result["errors"]
        return result
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}:{exc}")
        return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--account-id", choices=sorted(ACCOUNTS), required=True)
    ap.add_argument("--candidate-inventory", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = git_head(repo)
    if observed != args.analysis_source_git_commit:
        raise SystemExit(f"exact checkout mismatch: expected={args.analysis_source_git_commit}, observed={observed}")
    inventory_path = Path(args.candidate_inventory).resolve()
    inventory = load_json(inventory_path)
    if inventory.get("schema_version") != "1.0" or inventory.get("account_id") != args.account_id:
        raise SystemExit("candidate inventory schema/account mismatch")
    states = inventory.get("states") or {}
    unknown = set(states) - set(HISTORICAL_TRACKA_CLOSURE_SPECS)
    if unknown:
        raise SystemExit(f"artifact-availability candidate inventory may contain historical states only: {sorted(unknown)}")

    probed = {experiment_id: probe_state(experiment_id, states[experiment_id]) for experiment_id in sorted(states)}
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_v12_historical_artifact_availability",
        "account_id": args.account_id,
        "analysis_source_git_commit": observed,
        "candidate_inventory_sha256": sha256_file(inventory_path),
        "candidate_state_count": len(probed),
        "available_state_count": sum(row["available"] for row in probed.values()),
        "states": probed,
        "scientific_metrics_opened": False,
        "robustness_results_opened": False,
        "xai_results_opened": False,
        "selection_outcomes_opened": False,
        "placement_decision_authorized": True,
    }
    result["gate_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
