from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA, OperatorError, control_public_run_id, fetch_public_bundle, load_json,
    operator_runtime_head, publish_public_files, scientific_durable_map, write_json,
)
from master_attestations_v8 import verified_attestation_paths
from master_g2a_v8 import REQUIRED_G2A, owner_from_summary
from master_wait_v8 import POLL_SECONDS, dependency_wait_expired

CONTROL_FILES = [
    "TRACKA_V12_G2A_BARRIER.json", "TRACKA_V12_SCHEDULER_FREEZE.json",
    "TRACKA_V12_SCIENCE_GO.json", "TRACKA_V12_DURABLE_MAP.json",
    "TRACKA_V12_ACCOUNT_OWNERS.json", "TRACKA_V12_CONTROL_PUBLIC_REPORT.json",
]


def validate_control_bundle(repo: Path, g1a_seal: dict, control_dir: Path) -> dict:
    src = repo / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.tracka_v12_authorization import validate_science_authorization
    from cropcop_je.tracka_v12_g2a_v122 import validate_g2a_v122_barrier, validate_scheduler_freeze_v122
    from cropcop_je.tracka_v12_orchestration import validate_durable_map

    barrier = load_json(control_dir / "TRACKA_V12_G2A_BARRIER.json")
    scheduler = load_json(control_dir / "TRACKA_V12_SCHEDULER_FREEZE.json")
    go = load_json(control_dir / "TRACKA_V12_SCIENCE_GO.json")
    durable_map = load_json(control_dir / "TRACKA_V12_DURABLE_MAP.json")
    owners = load_json(control_dir / "TRACKA_V12_ACCOUNT_OWNERS.json")
    errors = validate_g2a_v122_barrier(barrier, expected_source_sha=SCIENCE_SHA, expected_g1a_seal_sha256=g1a_seal["g1a_seal_sha256"])
    errors += validate_scheduler_freeze_v122(scheduler, expected_g2a_barrier_sha256=barrier["barrier_sha256"])
    errors += validate_science_authorization(
        go, expected_source_sha=SCIENCE_SHA,
        expected_g1a_seal_sha256=g1a_seal["g1a_seal_sha256"],
        expected_g2a_barrier_sha256=barrier["barrier_sha256"],
        expected_scheduler_freeze_sha256=scheduler["scheduler_freeze_sha256"],
    )
    errors += validate_durable_map(durable_map)
    if set(owners) != {"K1", "K2", "K3"} or any(not str(v).strip() for v in owners.values()):
        errors.append("account owner map is incomplete")
    queues = scheduler.get("static_slot_queues") or {}
    for slot_id, queue in queues.items():
        account_id = str(slot_id).split("/", 1)[0]
        owner = str(owners.get(account_id, ""))
        for experiment_id in queue if isinstance(queue, list) else []:
            if not str(durable_map.get(experiment_id, "")).startswith(owner + "/"):
                errors.append(f"durable owner mismatch for {experiment_id}")
    if go.get("status") != "GO":
        errors.append("science GO status is not GO")
    if len(go.get("authorized_experiment_ids", [])) != 11:
        errors.append("science GO does not authorize exactly 11 states")
    if errors:
        raise OperatorError("control bundle validation failed: " + "; ".join(errors))
    return {"barrier": barrier, "scheduler": scheduler, "go": go, "durable_map": durable_map, "owners": owners}


def build_control_k1(repo: Path, *, g1a_bundle: Path, g1a_seal: dict, summaries: dict[str, Path], master_root: Path, stack: dict) -> tuple[Path, dict]:
    control_dir = master_root / "control"
    fetched = fetch_public_bundle(repo, control_public_run_id(), CONTROL_FILES, control_dir)
    if fetched is not None:
        control = validate_control_bundle(repo, g1a_seal, control_dir)
        print("Reusing already-published canonical Track-A control plane.")
        return control_dir, control
    shutil.rmtree(control_dir, ignore_errors=True); control_dir.mkdir(parents=True, exist_ok=True)
    barrier = control_dir / "TRACKA_V12_G2A_BARRIER.json"; scheduler = control_dir / "TRACKA_V12_SCHEDULER_FREEZE.json"
    command = [sys.executable, str(repo / "journal_extension/scripts/seal_tracka_v12_g2a.py")]
    for calibration_id in sorted(REQUIRED_G2A):
        command += ["--summary", str(summaries[calibration_id])]
    command += ["--barrier-out", str(barrier), "--scheduler-out", str(scheduler)]
    if subprocess.run(command, cwd=repo, text=True).returncode != 0:
        raise OperatorError("G2A barrier/scheduler sealer failed")

    code_attestation, lock_attestation = verified_attestation_paths()
    science_go = control_dir / "TRACKA_V12_SCIENCE_GO.json"
    command = [
        sys.executable, str(repo / "journal_extension/scripts/seal_tracka_v12_science_go_v123.py"),
        "--repo-root", str(repo), "--code-attestation", str(code_attestation),
        "--lock-runtime-attestation", str(lock_attestation), "--g1a-seal", str(g1a_bundle / "TRACKA_V12_G1A_SEAL.json"),
        "--g2a-barrier", str(barrier), "--scheduler-freeze", str(scheduler), "--output", str(science_go),
    ]
    if subprocess.run(command, cwd=repo, text=True).returncode != 0:
        raise OperatorError("final durability-bound SCIENCE_GO sealer failed")

    owners = {
        "K1": owner_from_summary(summaries["CAL-EFFB0"]),
        "K2": owner_from_summary(summaries["CAL-MNV4-LOGITS"]),
        "K3": owner_from_summary(summaries["CAL-R13"]),
    }
    if owner_from_summary(summaries["CAL-CNXTT"]) != owners["K1"]:
        raise OperatorError("K1 G2A summaries disagree on account owner")
    if owner_from_summary(summaries["CAL-MNV4-FEATURE"]) != owners["K2"]:
        raise OperatorError("K2 G2A summaries disagree on account owner")
    scheduler_payload = load_json(scheduler)
    durable_map = scientific_durable_map(scheduler_payload, owners)
    write_json(control_dir / "TRACKA_V12_DURABLE_MAP.json", durable_map)
    write_json(control_dir / "TRACKA_V12_ACCOUNT_OWNERS.json", owners)
    control = validate_control_bundle(repo, g1a_seal, control_dir)
    report = {
        "schema_version": "1.1", "stage": "TRACKA_V12_MASTER_CONTROL", "status": "PASS",
        "science_source_sha": SCIENCE_SHA, "operator_runtime_sha": operator_runtime_head(),
        "dependency_lock_sha256": stack["dependency_lock_sha256"], "g1a_seal_sha256": g1a_seal["g1a_seal_sha256"],
        "g2a_barrier_sha256": control["barrier"]["barrier_sha256"],
        "scheduler_freeze_sha256": control["scheduler"]["scheduler_freeze_sha256"],
        "science_authorization_sha256": control["go"]["authorization_sha256"],
        "authorized_experiment_count": 11, "account_owners": owners,
        "protected_test_accessed": False, "external_surface_accessed": False,
    }
    write_json(control_dir / "TRACKA_V12_CONTROL_PUBLIC_REPORT.json", report)
    publish_public_files(repo, control_public_run_id(), [control_dir / name for name in CONTROL_FILES])
    if fetch_public_bundle(repo, control_public_run_id(), CONTROL_FILES, control_dir) is None:
        raise OperatorError("control-plane publication did not round-trip through GitHub")
    return control_dir, validate_control_bundle(repo, g1a_seal, control_dir)


def acquire_control_worker(repo: Path, *, g1a_seal: dict, master_root: Path) -> tuple[Path, dict]:
    control_dir = master_root / "control"
    while True:
        fetched = fetch_public_bundle(repo, control_public_run_id(), CONTROL_FILES, control_dir)
        if fetched is not None:
            return control_dir, validate_control_bundle(repo, g1a_seal, control_dir)
        if dependency_wait_expired():
            raise TimeoutError("session dependency budget exhausted waiting for K1 canonical Track-A control plane")
        time.sleep(POLL_SECONDS)
