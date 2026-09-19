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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def run(command: list[str], *, cwd: Path) -> None:
    cp = subprocess.run(command, cwd=cwd, check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"preflight subprocess failed rc={cp.returncode}: {' '.join(command)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    analysis_sha = args.analysis_source_git_commit
    observed = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if observed != analysis_sha:
        raise SystemExit("preflight requires exact analysis checkout")

    spec_path = Path(args.spec).resolve()
    spec = load_json(spec_path)
    account = str(spec.get("account_id", ""))
    if account not in {"K1", "K2", "K3"} or spec.get("analysis_source_git_commit") != analysis_sha:
        raise SystemExit("preflight spec account/analysis-source mismatch")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    recovery_root = output / "recovery"
    recovery_root.mkdir()

    # Frozen campaign order: historical artifact availability is established
    # before terminal-metadata recovery. The historical probe is payload-shallow
    # and opens no scientific metrics.
    candidate_inventory = Path(spec["historical_candidate_inventory"]).resolve()
    availability = output / f"{account}_HISTORICAL_ARTIFACT_AVAILABILITY.json"
    run([
        os.environ.get("PYTHON", "python"),
        str(repo / "journal_extension/scripts/probe_tracka_v12_historical_artifacts.py"),
        "--repo-root", str(repo),
        "--account-id", account,
        "--candidate-inventory", str(candidate_inventory),
        "--analysis-source-git-commit", analysis_sha,
        "--output", str(availability),
    ], cwd=repo)
    availability_payload = load_json(availability)
    if availability_payload.get("status") != "PASS":
        raise SystemExit("historical artifact availability probe failed")
    if availability_payload.get("scientific_metrics_opened") is not False:
        raise SystemExit("historical availability probe opened scientific metrics before placement")

    recovery_rows = {}
    for row in spec.get("continuation_recoveries", []):
        experiment_id = row["experiment_id"]
        state_output = recovery_root / experiment_id
        run([
            os.environ.get("PYTHON", "python"),
            str(repo / "journal_extension/scripts/recover_tracka_v12_terminal_record.py"),
            "--repo-root", str(repo),
            "--experiment-id", experiment_id,
            "--run-id", row["run_id"],
            "--account-report", row["account_report"],
            "--checkpoint-root", row["checkpoint_root"],
            "--durable-store-locator", row["durable_store_locator"],
            "--output-dir", str(state_output),
        ], cwd=repo)
        cert = state_output / "TERMINAL_RECOVERY_CERTIFICATE.json"
        record = state_output / "RECOVERED_TERMINAL_RUN_RECORD.json"
        if not cert.is_file() or not record.is_file():
            raise SystemExit(f"terminal recovery did not produce required artifacts: {experiment_id}")
        recovery_rows[experiment_id] = {
            "recovered_run_record": str(record),
            "recovered_run_record_sha256": sha256_file(record),
            "recovery_certificate": str(cert),
            "recovery_certificate_sha256": sha256_file(cert),
        }

    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "manifest_kind": "track_a_posttraining_account_preflight",
        "account_id": account,
        "analysis_source_git_commit": analysis_sha,
        "spec_sha256": sha256_file(spec_path),
        "continuation_recovery_count": len(recovery_rows),
        "continuation_recoveries": recovery_rows,
        "historical_availability_report": str(availability),
        "historical_availability_report_sha256": sha256_file(availability),
        "scientific_metrics_opened": False,
        "robustness_results_opened": False,
        "xai_results_opened": False,
        "selection_outcomes_opened": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path = output / f"{account}_POSTTRAINING_PREFLIGHT_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
