from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tracka_v12_kaggle_operator_v2 import (
    SCIENCE_SHA,
    assert_clean_science_checkout,
    assert_kaggle_paths,
    assert_python_version,
    discover_g1a_bundle,
    ensure_private_dataset,
    ensure_science_checkout,
    install_locked_stack,
    load_json,
    load_kaggle_credentials,
    resolve_frozen_dataset,
    resolve_image_root,
    sanitized_child_env,
    validate_private_locators_with_science,
    verify_locked_stack,
    write_json,
)

VALID_ACCOUNTS = {"K1", "K2", "K3"}


def gpu_names() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def unique_control(name: str) -> Path:
    matches = sorted(path.resolve() for path in Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"{name} resolution must be unique, found {len(matches)}: {matches}")
    return matches[0]


def continuation_is_technical(summary: dict) -> bool:
    if summary.get("worker_errors"):
        return False
    for slot_rows in (summary.get("slot_results") or {}).values():
        for row in slot_rows:
            if row.get("return_code") not in {0, None}:
                return False
            if row.get("run_status") == "FAIL" or row.get("slot_quarantined") is True:
                return False
    return True


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_ACCOUNTS:
        raise SystemExit("usage: science_account_driver.py K1|K2|K3")
    account_id = sys.argv[1]

    assert_kaggle_paths()
    assert_python_version()

    output_root = Path(f"/kaggle/working/TRACKA_V12_SCIENCE_{account_id}")
    output_root.mkdir(parents=True, exist_ok=True)

    repo = ensure_science_checkout()
    install_locked_stack(repo)
    stack = verify_locked_stack(repo)
    assert_clean_science_checkout(repo)

    names = gpu_names()
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
        raise RuntimeError(f"scientific account requires qualified T4x2; observed {names}")

    manifest, class_map = resolve_frozen_dataset(
        manifest_override=os.environ.get("CROPCOP_MANIFEST", ""),
        class_map_override=os.environ.get("CROPCOP_CLASS_MAP", ""),
    )
    image_root = resolve_image_root(
        manifest,
        override=os.environ.get("CROPCOP_IMAGE_ROOT", ""),
    )
    g1a_bundle = discover_g1a_bundle(
        override=os.environ.get("CROPCOP_G1A_BUNDLE", ""),
    )

    g2a_barrier = unique_control("TRACKA_V12_G2A_BARRIER.json")
    scheduler_path = unique_control("TRACKA_V12_SCHEDULER_FREEZE.json")
    science_go = unique_control("TRACKA_V12_SCIENCE_GO.json")
    durable_map_path = unique_control("TRACKA_V12_DURABLE_MAP.json")

    g1a = load_json(g1a_bundle / "TRACKA_V12_G1A_SEAL.json")
    g2a = load_json(g2a_barrier)
    scheduler = load_json(scheduler_path)
    durable_map = load_json(durable_map_path)
    go = load_json(science_go)

    sys.path.insert(0, str(repo / "journal_extension/src"))
    from cropcop_je.tracka_v12_authorization import validate_science_authorization
    from cropcop_je.tracka_v12_g2a_v122 import validate_g2a_v122_barrier, validate_scheduler_freeze_v122
    from cropcop_je.tracka_v12_orchestration import validate_durable_map

    g2_errors = validate_g2a_v122_barrier(
        g2a,
        expected_source_sha=SCIENCE_SHA,
        expected_g1a_seal_sha256=g1a["g1a_seal_sha256"],
    )
    if g2_errors:
        raise RuntimeError("G2A barrier invalid: " + "; ".join(g2_errors))
    scheduler_errors = validate_scheduler_freeze_v122(
        scheduler,
        expected_g2a_barrier_sha256=g2a["barrier_sha256"],
    )
    if scheduler_errors:
        raise RuntimeError("scheduler freeze invalid: " + "; ".join(scheduler_errors))
    auth_errors = validate_science_authorization(
        go,
        expected_source_sha=SCIENCE_SHA,
        expected_g1a_seal_sha256=g1a["g1a_seal_sha256"],
        expected_g2a_barrier_sha256=g2a["barrier_sha256"],
        expected_scheduler_freeze_sha256=scheduler["scheduler_freeze_sha256"],
    )
    if auth_errors:
        raise RuntimeError("science GO invalid: " + "; ".join(auth_errors))
    if go.get("status") != "GO":
        raise RuntimeError("science GO is not GO")

    map_errors = validate_durable_map(durable_map)
    if map_errors:
        raise RuntimeError("full scientific durable map invalid: " + "; ".join(map_errors))

    queues = scheduler.get("static_slot_queues") or {}
    slot_ids = [f"{account_id}/GPU0", f"{account_id}/GPU1"]
    assigned: list[str] = []
    for slot_id in slot_ids:
        queue = queues.get(slot_id)
        if not isinstance(queue, list) or not queue:
            raise RuntimeError(f"sealed scheduler queue missing/empty for {slot_id}")
        assigned.extend(queue)

    username, key = load_kaggle_credentials()
    assigned_map = {experiment_id: durable_map[experiment_id] for experiment_id in assigned}
    wrong_owner = {
        experiment_id: locator
        for experiment_id, locator in assigned_map.items()
        if locator.split("/", 1)[0].casefold() != username.casefold()
    }
    if wrong_owner:
        raise RuntimeError(
            "authenticated Kaggle account does not match the sealed durable-map owner for this lane: "
            + json.dumps(wrong_owner, indent=2)
        )

    kaggle_env = dict(os.environ)
    for experiment_id, locator in assigned_map.items():
        ensure_private_dataset(locator, env=kaggle_env)
    durable_preflight = validate_private_locators_with_science(
        repo,
        assigned_map,
        env=kaggle_env,
    )

    parent_env = sanitized_child_env()
    parent_env["KAGGLE_USERNAME"] = username
    parent_env["KAGGLE_KEY"] = key

    command = [
        sys.executable,
        str(repo / "journal_extension/kaggle/run_tracka_v12_account.py"),
        "--account-id", account_id,
        "--repo-root", str(repo),
        "--source-git-commit", SCIENCE_SHA,
        "--manifest", str(manifest),
        "--class-map", str(class_map),
        "--image-root", str(image_root),
        "--g1a-bundle", str(g1a_bundle),
        "--g2a-barrier", str(g2a_barrier),
        "--scheduler-freeze", str(scheduler_path),
        "--science-authorization", str(science_go),
        "--durable-map", str(durable_map_path),
        "--output-root", str(output_root),
        "--num-workers", "4",
        "--checkpoint-every-steps", "250",
        "--session-hard-limit-seconds", str(12 * 3600),
        "--finalization-margin-seconds", "3600",
        "--minimum-new-run-safe-seconds", "1800",
        "--min-free-gb", "5",
    ]

    log_path = output_root / f"{account_id}_ACCOUNT_CONSOLE.log"
    with log_path.open("a", encoding="utf-8") as log:
        cp = subprocess.run(
            command,
            cwd=repo,
            env=parent_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    summary_path = output_root / account_id / "ACCOUNT_EXECUTION_SUMMARY.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    if cp.returncode not in {0, 2}:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"account runner failed rc={cp.returncode}\n{tail}")
    if cp.returncode == 2 and not continuation_is_technical(summary):
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            "account runner returned ATTENTION_REQUIRED for a non-rollover technical failure; "
            "do not blindly resume.\n" + tail
        )

    report = {
        "schema_version": "1.1",
        "stage": "TRACKA_V12_SCIENCE_ACCOUNT",
        "account_id": account_id,
        "science_source_sha": SCIENCE_SHA,
        "science_go_status": go["status"],
        "dependency_lock_sha256": stack["dependency_lock_sha256"],
        "runner_return_code": cp.returncode,
        "runner_status": summary.get("status"),
        "science_complete": summary.get("science_complete", False),
        "assigned_states": assigned,
        "durable_locators": assigned_map,
        "durability_preflight_status": durable_preflight["status"],
        "gpu_names": names,
        "child_git_credentials_removed": summary.get("child_git_credentials_removed"),
        "cross_gpu_gradient_synchronization": summary.get("cross_gpu_gradient_synchronization"),
    }
    write_json(output_root / f"{account_id}_OPERATOR_REPORT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))

    if cp.returncode == 0:
        if summary.get("status") != "PASS" or summary.get("science_complete") is not True:
            raise RuntimeError("runner returned success without terminal science completion")
        print("ACCOUNT TERMINAL PASS")
        return 0

    print(
        "SESSION ROLLOVER REQUIRED. Start a fresh Kaggle session with the same inputs/account and rerun "
        "this notebook. Canonical run IDs plus private durable checkpoints will restore the same scientific state."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
