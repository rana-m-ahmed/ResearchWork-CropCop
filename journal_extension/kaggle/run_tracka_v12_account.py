from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.envelope import gpu_inventory, validate_t4x2_inventory
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source
from cropcop_je.tracka_v12_authorization import validate_science_authorization
from cropcop_je.tracka_v12_g2a_v122 import validate_g2a_v122_barrier, validate_scheduler_freeze_v122
from cropcop_je.tracka_v12_orchestration import (
    ACCOUNT_SLOTS,
    account_output_path,
    account_queue_manifest,
    git_credentials_present,
    public_evidence_candidates,
    sanitized_child_environment,
    scientific_run_id,
    validate_durable_map,
)
from cropcop_je.tracka_v12_runtime import load_and_validate_g1a_bundle

RUNNER = SCRIPTS / "run_tracka_v12_training_v121.py"
PUBLICATION_LOCK = threading.Lock()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-id", choices=tuple(ACCOUNT_SLOTS), required=True)
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--source-git-commit", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--g1a-bundle", required=True)
    ap.add_argument("--g2a-barrier", required=True)
    ap.add_argument("--scheduler-freeze", required=True)
    ap.add_argument("--science-authorization", required=True)
    ap.add_argument("--durable-map", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--checkpoint-every-steps", type=int, default=250)
    ap.add_argument("--session-hard-limit-seconds", type=float, default=12 * 3600)
    ap.add_argument("--finalization-margin-seconds", type=float, default=3600)
    ap.add_argument("--minimum-new-run-safe-seconds", type=float, default=1800)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--publish-evidence", action="store_true")
    return ap


def preflight(args) -> tuple[dict, dict, dict, dict, SessionBudget]:
    repo = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    verify_clean_source(repo, authorized_source_sha=args.source_git_commit, output_roots=[output_root])

    inventory = gpu_inventory()
    gpu_errors = validate_t4x2_inventory(inventory)
    if gpu_errors:
        raise RuntimeError("account T4x2 preflight failed: " + "; ".join(gpu_errors))

    g1a, g1a_errors = load_and_validate_g1a_bundle(
        args.g1a_bundle,
        expected_source_sha=args.source_git_commit,
    )
    if g1a_errors:
        raise RuntimeError("account G1A preflight failed: " + "; ".join(g1a_errors))
    g1a_sha = str(g1a.get("g1a_seal_sha256", ""))

    g2a = load_json(args.g2a_barrier)
    g2_errors = validate_g2a_v122_barrier(
        g2a,
        expected_source_sha=args.source_git_commit,
        expected_g1a_seal_sha256=g1a_sha,
    )
    if g2_errors:
        raise RuntimeError("account G2A preflight failed: " + "; ".join(g2_errors))

    scheduler = load_json(args.scheduler_freeze)
    scheduler_errors = validate_scheduler_freeze_v122(
        scheduler,
        expected_g2a_barrier_sha256=g2a["barrier_sha256"],
    )
    if scheduler_errors:
        raise RuntimeError("account scheduler preflight failed: " + "; ".join(scheduler_errors))
    if scheduler.get("source_git_commit") != args.source_git_commit:
        raise RuntimeError("account scheduler source SHA mismatch")
    if scheduler.get("g1a_seal_sha256") != g1a_sha:
        raise RuntimeError("account scheduler/G1A binding mismatch")

    authorization = load_json(args.science_authorization)
    auth_errors = validate_science_authorization(
        authorization,
        expected_source_sha=args.source_git_commit,
        expected_g1a_seal_sha256=g1a_sha,
        expected_g2a_barrier_sha256=g2a["barrier_sha256"],
        expected_scheduler_freeze_sha256=scheduler["scheduler_freeze_sha256"],
    )
    if auth_errors:
        raise RuntimeError("account science authorization failed: " + "; ".join(auth_errors))

    durable_map = load_json(args.durable_map)
    durable_errors = validate_durable_map(durable_map)
    if durable_errors:
        raise RuntimeError("account durable map invalid: " + "; ".join(durable_errors))

    account_manifest = account_queue_manifest(scheduler, args.account_id)
    if set(account_manifest.get("slots", {})) != set(ACCOUNT_SLOTS[args.account_id]):
        raise RuntimeError("account frozen queue does not match the two qualified physical slots")
    budget = SessionBudget.establish_global_clock(
        hard_limit_seconds=args.session_hard_limit_seconds,
        finalization_margin_seconds=args.finalization_margin_seconds,
    )
    return inventory, scheduler, authorization, durable_map, budget


def child_command(args, *, experiment_id: str, slot_id: str, run_id: str, output_dir: Path, durable_locator: str) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--repo-root", str(Path(args.repo_root).resolve()),
        "--experiment-id", experiment_id,
        "--manifest", str(Path(args.manifest).resolve()),
        "--class-map", str(Path(args.class_map).resolve()),
        "--image-root", str(Path(args.image_root).resolve()),
        "--g1a-bundle", str(Path(args.g1a_bundle).resolve()),
        "--g2a-barrier", str(Path(args.g2a_barrier).resolve()),
        "--scheduler-freeze", str(Path(args.scheduler_freeze).resolve()),
        "--science-authorization", str(Path(args.science_authorization).resolve()),
        "--dependency-lock", args.dependency_lock,
        "--run-id", run_id,
        "--slot-id", slot_id,
        "--source-git-commit", args.source_git_commit,
        "--output-dir", str(output_dir),
        "--num-workers", str(args.num_workers),
        "--checkpoint-every-steps", str(args.checkpoint_every_steps),
        "--resume-mode", "auto",
        "--mode", "scientific",
        "--session-hard-limit-seconds", str(args.session_hard_limit_seconds),
        "--finalization-margin-seconds", str(args.finalization_margin_seconds),
        "--min-free-gb", str(args.min_free_gb),
        "--durable-store-kind", "kaggle-dataset",
        "--durable-store-locator", durable_locator,
        "--durable-required",
    ]


def publish_run(args, *, run_id: str, output_dir: Path) -> dict:
    if not args.publish_evidence:
        return {"status": "NOT_REQUESTED", "branch": None, "error": None}
    files = public_evidence_candidates(output_dir)
    try:
        with PUBLICATION_LOCK:
            branch = publish_to_github_branch(
                repo_dir=Path(args.repo_root).resolve(),
                source_git_sha=args.source_git_commit,
                run_id=run_id,
                files=files,
            )
        return {"status": "PASS", "branch": branch, "error": None}
    except Exception as exc:
        return {"status": "REPAIR_REQUIRED", "branch": None, "error": f"{type(exc).__name__}: {exc}"}


def worker(args, *, slot_id: str, queue: list[str], durable_map: dict, budget: SessionBudget, results: dict, lock: threading.Lock) -> None:
    slot_results = []
    for experiment_id in queue:
        if budget.remaining_safe_seconds <= float(args.minimum_new_run_safe_seconds):
            slot_results.append(
                {
                    "experiment_id": experiment_id,
                    "status": "NOT_STARTED_SESSION_BUDGET",
                    "remaining_safe_seconds": budget.remaining_safe_seconds,
                }
            )
            break
        run_id = scientific_run_id(experiment_id, args.source_git_commit)
        output_dir = account_output_path(
            args.output_root,
            account_id=args.account_id,
            experiment_id=experiment_id,
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        env = sanitized_child_environment(dict(os.environ), slot_id=slot_id)
        if git_credentials_present(env):
            raise RuntimeError("child environment still contains Git credentials after sanitization")
        command = child_command(
            args,
            experiment_id=experiment_id,
            slot_id=slot_id,
            run_id=run_id,
            output_dir=output_dir,
            durable_locator=durable_map[experiment_id],
        )
        log_path = output_dir / "console.log"
        with log_path.open("a", encoding="utf-8") as log:
            cp = subprocess.run(
                command,
                cwd=Path(args.repo_root).resolve(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record_path = output_dir / "run_record.json"
        record = load_json(record_path) if record_path.is_file() else {}
        publication = publish_run(args, run_id=run_id, output_dir=output_dir) if record.get("status") == "PASS" else {"status": "NOT_TERMINAL", "branch": None, "error": None}
        result = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "slot_id": slot_id,
            "return_code": cp.returncode,
            "run_status": record.get("status"),
            "continuation_required": record.get("continuation_required"),
            "selected_checkpoint_sha256": (record.get("result_summary") or {}).get("selected_checkpoint_sha256"),
            "publication": publication,
            "console_log": str(log_path),
        }
        slot_results.append(result)
        if cp.returncode != 0 or record.get("status") == "FAIL":
            result["slot_quarantined"] = True
            break
        if record.get("continuation_required") is True or record.get("status") == "LAUNCHED":
            result["session_rollover_required"] = True
            break
        if record.get("status") != "PASS":
            result["slot_quarantined"] = True
            break
    with lock:
        results[slot_id] = slot_results


def main() -> int:
    args = parser().parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    account_summary_path = output_root / args.account_id / "ACCOUNT_EXECUTION_SUMMARY.json"
    account_summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        inventory, scheduler, authorization, durable_map, budget = preflight(args)
        account_manifest = account_queue_manifest(scheduler, args.account_id)
        atomic_write_json(account_summary_path.parent / "FROZEN_ACCOUNT_QUEUE.json", account_manifest)
        results: dict[str, list[dict]] = {}
        lock = threading.Lock()
        threads = []
        for slot_id in ACCOUNT_SLOTS[args.account_id]:
            queue = list(account_manifest["slots"][slot_id]["queue"])
            thread = threading.Thread(
                target=worker,
                kwargs={
                    "args": args,
                    "slot_id": slot_id,
                    "queue": queue,
                    "durable_map": durable_map,
                    "budget": budget,
                    "results": results,
                    "lock": lock,
                },
                name=f"tracka-{slot_id.replace('/', '-')}",
                daemon=False,
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        summary = {
            "schema_version": "1.0",
            "status": "PASS" if all(
                all(row.get("run_status") == "PASS" for row in results.get(slot, []))
                for slot in ACCOUNT_SLOTS[args.account_id]
            ) else "ATTENTION_REQUIRED",
            "account_id": args.account_id,
            "source_git_commit": args.source_git_commit,
            "scheduler_freeze_sha256": scheduler["scheduler_freeze_sha256"],
            "science_authorization_sha256": authorization["authorization_sha256"],
            "gpu_inventory": inventory,
            "session_budget": budget.snapshot(),
            "slot_results": results,
            "child_git_credentials_removed": True,
            "one_child_per_visible_gpu": True,
            "cross_gpu_gradient_synchronization": False,
        }
        atomic_write_json(account_summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 2
    except BaseException as exc:
        failure = {
            "schema_version": "1.0",
            "status": "FAIL",
            "account_id": args.account_id,
            "source_git_commit": args.source_git_commit,
            "failure": {
                "type": type(exc).__name__,
                "reason": str(exc),
                "traceback_tail": traceback.format_exc()[-5000:],
            },
        }
        atomic_write_json(account_summary_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
