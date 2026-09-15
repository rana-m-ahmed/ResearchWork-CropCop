from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA,
    OperatorError,
    account_public_run_id,
    ensure_private_dataset,
    load_json,
    operator_runtime_head,
    technical_rollover_only,
    validate_private_locators_with_science,
    wait_kaggle_dataset_ready,
    write_json,
)
from master_publication_v8 import publish_public_files


def ensure_science_durability(
    repo: Path, *, account_id: str, username: str, kaggle_env: dict[str, str], control: dict,
) -> dict[str, str]:
    queues = control["scheduler"]["static_slot_queues"]
    assigned: list[str] = []
    for slot_id in (f"{account_id}/GPU0", f"{account_id}/GPU1"):
        queue = queues.get(slot_id)
        if not isinstance(queue, list) or not queue:
            raise OperatorError(f"missing frozen scheduler queue for {slot_id}")
        assigned.extend(queue)
    durable_map = control["durable_map"]
    mapping = {experiment_id: durable_map[experiment_id] for experiment_id in assigned}
    wrong_owner = {
        eid: locator for eid, locator in mapping.items()
        if locator.split("/", 1)[0].casefold() != username.casefold()
    }
    if wrong_owner:
        raise OperatorError(
            "authenticated Kaggle account does not own its frozen scientific durability targets: "
            + json.dumps(wrong_owner, sort_keys=True)
        )
    for locator in mapping.values():
        ensure_private_dataset(locator, env=kaggle_env)
        wait_kaggle_dataset_ready(locator, env=kaggle_env)
    validate_private_locators_with_science(repo, mapping, env=kaggle_env)
    return mapping


def science_command(
    repo: Path,
    *,
    account_id: str,
    manifest: Path,
    class_map: Path,
    image_root: Path,
    g1a_bundle: Path,
    control_dir: Path,
    master_root: Path,
) -> list[str]:
    """Invoke the exact qualified v1.2.1 parent runner; publication remains parent-only."""
    return [
        sys.executable,
        str(repo / "journal_extension/kaggle/run_tracka_v12_account_v121.py"),
        "--account-id", account_id,
        "--repo-root", str(repo),
        "--source-git-commit", SCIENCE_SHA,
        "--manifest", str(manifest),
        "--class-map", str(class_map),
        "--image-root", str(image_root),
        "--g1a-bundle", str(g1a_bundle),
        "--g2a-barrier", str(control_dir / "TRACKA_V12_G2A_BARRIER.json"),
        "--scheduler-freeze", str(control_dir / "TRACKA_V12_SCHEDULER_FREEZE.json"),
        "--science-authorization", str(control_dir / "TRACKA_V12_SCIENCE_GO.json"),
        "--durable-map", str(control_dir / "TRACKA_V12_DURABLE_MAP.json"),
        "--output-root", str(master_root / "science"),
        "--num-workers", "4",
        "--checkpoint-every-steps", "250",
        "--session-hard-limit-seconds", str(12 * 3600),
        "--finalization-margin-seconds", "3600",
        "--minimum-new-run-safe-seconds", "1800",
        "--min-free-gb", "5",
    ]


def _orchestration_api(repo: Path):
    src = repo / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.tracka_v12_orchestration import account_output_path, public_evidence_candidates
    return account_output_path, public_evidence_candidates


def publish_completed_scientific_runs(repo: Path, account_id: str, summary: dict, master_root: Path) -> dict[str, dict]:
    account_output_path, public_evidence_candidates = _orchestration_api(repo)
    results: dict[str, dict] = {}
    output_root = master_root / "science"
    for slot_id, rows in (summary.get("slot_results") or {}).items():
        if not isinstance(rows, list):
            raise OperatorError(f"malformed slot_results for {slot_id}")
        for row in rows:
            if not isinstance(row, dict):
                raise OperatorError(f"malformed scientific row for {slot_id}")
            experiment_id = str(row.get("experiment_id") or "")
            run_id = str(row.get("run_id") or "")
            if not experiment_id or not run_id:
                continue
            if row.get("return_code") != 0 or row.get("run_status") != "PASS":
                continue
            run_dir = account_output_path(output_root, experiment_id)
            candidates = list(public_evidence_candidates(run_dir))
            if not candidates:
                raise OperatorError(f"terminal PASS scientific run has no public evidence candidates: {experiment_id}")
            branch = publish_public_files(repo, run_id, candidates)
            results[experiment_id] = {
                "run_id": run_id,
                "status": "PASS",
                "branch": branch,
                "file_count": len(candidates),
            }
    return results


def sanitized_account_report(account_id: str, summary: dict, control: dict, publications: dict[str, dict]) -> dict:
    slots = {}
    for slot_id, rows in (summary.get("slot_results") or {}).items():
        slots[slot_id] = [
            {
                "experiment_id": row.get("experiment_id"),
                "run_id": row.get("run_id"),
                "return_code": row.get("return_code"),
                "run_status": row.get("run_status"),
                "continuation_required": row.get("continuation_required"),
                "selected_checkpoint_sha256": row.get("selected_checkpoint_sha256"),
                "operator_publication": publications.get(str(row.get("experiment_id") or "")),
            }
            for row in rows
        ]
    return {
        "schema_version": "1.2.1",
        "stage": "TRACKA_V12_MASTER_ACCOUNT",
        "account_id": account_id,
        "status": summary.get("status"),
        "science_complete": summary.get("science_complete"),
        "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(),
        "scheduler_freeze_sha256": control["scheduler"]["scheduler_freeze_sha256"],
        "science_authorization_sha256": control["go"]["authorization_sha256"],
        "slot_results": slots,
        "execution_errors": summary.get("execution_errors", []),
        "operator_publication_count": len(publications),
        "scientific_runner_git_publication_requested": False,
        "child_git_credentials_removed": summary.get("child_git_credentials_removed"),
        "cross_gpu_gradient_synchronization": summary.get("cross_gpu_gradient_synchronization"),
    }


def run_science(
    repo: Path,
    *,
    account_id: str,
    manifest: Path,
    class_map: Path,
    image_root: Path,
    g1a_bundle: Path,
    control_dir: Path,
    control: dict,
    master_root: Path,
) -> int:
    command = science_command(
        repo,
        account_id=account_id,
        manifest=manifest,
        class_map=class_map,
        image_root=image_root,
        g1a_bundle=g1a_bundle,
        control_dir=control_dir,
        master_root=master_root,
    )
    cp = subprocess.run(command, cwd=repo, env=dict(os.environ), text=True)
    summary_path = master_root / "science" / account_id / "ACCOUNT_EXECUTION_SUMMARY.json"
    if not summary_path.is_file():
        raise OperatorError(f"scientific account runner produced no summary (rc={cp.returncode})")
    summary = load_json(summary_path)

    publication_error: Exception | None = None
    publications: dict[str, dict] = {}
    try:
        publications = publish_completed_scientific_runs(repo, account_id, summary, master_root)
    except Exception as exc:
        publication_error = exc
        print(f"Scientific public evidence publication requires repair: {type(exc).__name__}: {exc}")

    public_report = master_root / f"TRACKA_V12_{account_id}_PUBLIC_REPORT.json"
    write_json(public_report, sanitized_account_report(account_id, summary, control, publications))
    account_report_error: Exception | None = None
    try:
        publish_public_files(repo, account_public_run_id(account_id), [public_report])
    except Exception as exc:
        account_report_error = exc
        print(f"Account-level public report publication requires repair: {type(exc).__name__}: {exc}")

    if cp.returncode == 0:
        if summary.get("status") != "PASS" or summary.get("science_complete") is not True:
            raise OperatorError("account runner returned success without terminal science PASS")
        if publication_error is not None or account_report_error is not None:
            print(f"{account_id}: science complete; GitHub publication needs repair. Rerun SAME master notebook.")
            return 2
        print(f"{account_id}: SCIENTIFIC QUEUE TERMINAL PASS")
        return 0

    if cp.returncode == 2 and technical_rollover_only(summary):
        if publication_error is not None or account_report_error is not None:
            print(f"{account_id}: technical rollover with publication repair pending. Rerun SAME master notebook.")
        else:
            print(f"{account_id}: planned session rollover. Rerun this SAME master notebook in a fresh Batch session.")
        return 2

    raise OperatorError(
        f"{account_id}: scientific account runner requires investigation; rc={cp.returncode}, "
        f"status={summary.get('status')}, errors={summary.get('execution_errors')}"
    )
