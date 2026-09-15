from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    MASTER_ACCOUNTS,
    SCIENCE_SHA,
    OperatorError,
    assert_clean_science_checkout,
    assert_kaggle_batch,
    assert_kaggle_paths,
    assert_python_version,
    ensure_locked_stack,
    ensure_science_checkout,
    github_write_preflight,
    kaggle_dataset_exists,
    load_github_token,
    load_kaggle_credentials,
    operator_runtime_head,
    sha256_file,
)
from master_attestations_v8 import materialize_verified_attestations, verified_release_attestation_manifest
from master_preflight import resolve_master_inputs
from master_g1a_v8 import acquire_canonical_g1a_worker, ensure_canonical_g1a_k1
from master_g2a_v8 import ensure_account_g2a
from master_control_v8 import build_control_k1
from master_continuation_v11 import probe_current_g1a_handoff, try_acquire_control_worker, try_collect_all_g2a
from master_publication_v8 import install_stage_publication_hooks
from master_science_v8 import ensure_science_durability, run_science

HEARTBEAT_SECONDS = 60.0
_CURRENT_STAGE = "BOOT"
_STAGE_LOCK = threading.Lock()


def guard_identity(account_id: str) -> str:
    token = str(os.environ.get("CROPCOP_MASTER_GUARD_TOKEN", "") or "").strip()
    guard_account = str(os.environ.get("CROPCOP_MASTER_GUARD_ACCOUNT", "") or "").strip()
    lock_path = str(os.environ.get("CROPCOP_MASTER_GUARD_LOCK", "") or "").strip()
    if len(token) < 16 or guard_account != account_id or not lock_path:
        raise OperatorError("v11 master driver must be launched through master_launch_guard_v11.py")
    return token


def stage(name: str, account_id: str, guard_token: str) -> None:
    global _CURRENT_STAGE
    with _STAGE_LOCK:
        _CURRENT_STAGE = name
    print(
        f"\n=== TRACKA_V12_MASTER_V11 {account_id} :: {name} "
        f"pid={os.getpid()} ppid={os.getppid()} launch={guard_token[:12]} "
        f"science={SCIENCE_SHA[:12]} runtime={operator_runtime_head()[:12]} ===",
        flush=True,
    )


def _heartbeat(stop: threading.Event, account_id: str, guard_token: str, started: float) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        with _STAGE_LOCK:
            name = _CURRENT_STAGE
        elapsed = int(time.monotonic() - started)
        print(
            f"MASTER_HEARTBEAT_V11 account={account_id} stage={name} elapsed={elapsed}s "
            f"pid={os.getpid()} launch={guard_token[:12]} science={SCIENCE_SHA[:12]} "
            f"runtime={operator_runtime_head()[:12]}",
            flush=True,
        )


def gpu_names() -> list[str]:
    cp = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def assert_t4x2() -> list[str]:
    names = gpu_names()
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
        raise OperatorError(f"master notebook requires Kaggle T4 x2; observed {names}")
    return names


def assert_expected_account(account_id: str, username: str) -> None:
    expected = str(os.environ.get("CROPCOP_EXPECTED_KAGGLE_USERNAME", "") or "").strip()
    if not expected:
        raise OperatorError(
            f"{account_id} requires CROPCOP_EXPECTED_KAGGLE_USERNAME to be set by the account-specific notebook; "
            "refusing an unbound Kaggle identity"
        )
    if username.casefold() != expected.casefold():
        raise OperatorError(f"{account_id} Kaggle account mismatch: expected {expected!r}, credential username is {username!r}")
    print(f"KAGGLE_ACCOUNT_BINDING {account_id}={username}", flush=True)


def verify_release_integrity() -> dict:
    manifest_path, manifest = verified_release_attestation_manifest()
    return {
        "science_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(),
        "release_attestation_manifest_sha256": sha256_file(manifest_path),
        "code_attestation_member_sha256": manifest["code_attestation"]["member_sha256"],
        "lock_runtime_attestation_member_sha256": manifest["lock_runtime_attestation"]["member_sha256"],
        "r13_parity_contract_id": manifest["r13_parity_contract"]["contract_id"],
        "r13_parity_required_max_abs_difference": manifest["r13_parity_contract"]["required_max_abs_difference"],
        "remaining_scientific_states": manifest["remaining_scientific_states"],
    }


def retry_operator_call(label: str, fn, *, attempts: int = 3):
    if attempts < 1:
        raise OperatorError("retry attempts must be positive")
    last_error: Exception | None = None
    delays = (5, 15, 30)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (OperatorError, subprocess.SubprocessError, OSError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = delays[min(attempt - 1, len(delays) - 1)]
            print(f"{label} attempt {attempt}/{attempts} failed: {type(exc).__name__}; retrying in {delay}s", flush=True)
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def controlled_dependency_continuation(account_id: str, reason: str) -> int:
    print(f"CONTROLLED_DEPENDENCY_CONTINUATION {account_id}: {reason}", flush=True)
    print("End this Batch session and rerun the SAME v11 notebook after the prerequisite is ready.", flush=True)
    return 2


def _run_account(account_id: str, guard_token: str, global_clock: float) -> int:
    stage("RELEASE_INTEGRITY_PREFLIGHT", account_id, guard_token)
    release = verify_release_integrity()
    print("RELEASE_INTEGRITY_PASS", release, flush=True)

    stage("RUNTIME_AND_HARDWARE_PREFLIGHT", account_id, guard_token)
    assert_kaggle_paths(); assert_kaggle_batch(); assert_python_version()
    print("GPU inventory:", assert_t4x2(), flush=True)
    username, key = load_kaggle_credentials()
    assert_expected_account(account_id, username)
    load_github_token()
    kaggle_env = dict(os.environ); kaggle_env["KAGGLE_USERNAME"] = username; kaggle_env["KAGGLE_KEY"] = key

    master_root = Path(f"/kaggle/working/TRACKA_V12_MASTER_{account_id}")
    master_root.mkdir(parents=True, exist_ok=True)

    stage("FROZEN_INPUT_RESOLUTION", account_id, guard_token)
    manifest, class_map, image_root = resolve_master_inputs(account_id)
    print("BOUND_MANIFEST=", manifest); print("BOUND_CLASS_MAP=", class_map); print("BOUND_IMAGE_ROOT=", image_root)
    if account_id == "K1": print("BOUND_PRINCIPAL_G1=", os.environ.get("CROPCOP_PRINCIPAL_G1", ""))

    stage("SCIENCE_SOURCE_GITHUB_AND_ATTESTATION_PREFLIGHT", account_id, guard_token)
    repo = retry_operator_call("Frozen science checkout", ensure_science_checkout)
    assert_clean_science_checkout(repo)
    retry_operator_call("GitHub evidence write preflight", lambda: github_write_preflight(repo))
    code_attestation, lock_attestation = retry_operator_call(
        "Exact-head packaged attestation materialization",
        lambda: materialize_verified_attestations(master_root / "release-attestations"),
    )
    print(
        "EXACT_HEAD_ATTESTATIONS_PASS",
        {"code": sha256_file(code_attestation), "lock_runtime": sha256_file(lock_attestation)},
        flush=True,
    )
    assert_clean_science_checkout(repo)

    worker_handoff = None
    if account_id != "K1":
        stage("WORKER_G1A_PREREQUISITE_PROBE", account_id, guard_token)
        worker_handoff = probe_current_g1a_handoff(
            repo,
            destination=master_root / "handoff-probe" / "TRACKA_V12_G1A_HANDOFF.json",
        )
        if worker_handoff is None:
            return controlled_dependency_continuation(
                account_id,
                "exact-runtime K1 G1A READY handoff is not yet published; skipping expensive stack installation",
            )
        print(
            f"WORKER_G1A_PREREQUISITE_READY locator={worker_handoff['private_kaggle_dataset_locator']} "
            f"seal={str(worker_handoff['g1a_seal_sha256'])[:12]}",
            flush=True,
        )

    stage("EXACT_EXECUTION_STACK", account_id, guard_token)
    stack = ensure_locked_stack(repo); assert_clean_science_checkout(repo)
    print("DEPENDENCY_LOCK_SHA256=", stack["dependency_lock_sha256"], flush=True)
    install_stage_publication_hooks()

    stage("CANONICAL_G1A", account_id, guard_token)
    if account_id == "K1":
        g1a_bundle, g1a_seal, shared_locator = ensure_canonical_g1a_k1(
            repo, manifest=manifest, class_map=class_map, image_root=image_root,
            username=username, kaggle_env=kaggle_env, stack=stack, master_root=master_root,
        )
    else:
        assert worker_handoff is not None
        expected_locator = str(worker_handoff["private_kaggle_dataset_locator"])
        if not kaggle_dataset_exists(expected_locator, env=kaggle_env):
            raise OperatorError(
                f"K1 G1A handoff is READY but private dataset {expected_locator!r} is not readable by {username!r}. "
                "Grant this Kaggle account Can view access before rerunning."
            )
        g1a_bundle, g1a_seal, shared_locator = acquire_canonical_g1a_worker(
            repo, kaggle_env=kaggle_env, master_root=master_root
        )
    print(f"{account_id}: canonical G1A PASS {g1a_seal['g1a_seal_sha256']} via {shared_locator}", flush=True)

    stage("ACCOUNT_G2A", account_id, guard_token)
    ensure_account_g2a(
        repo, account_id=account_id, username=username, kaggle_env=kaggle_env,
        manifest=manifest, class_map=class_map, image_root=image_root,
        g1a_bundle=g1a_bundle, g1a_seal=g1a_seal, master_root=master_root, global_clock=global_clock,
    )
    print(f"{account_id}: account G2A evidence canonicalized.", flush=True)

    stage("CONTROL_PLANE", account_id, guard_token)
    if account_id == "K1":
        summaries = try_collect_all_g2a(repo, g1a_seal=g1a_seal, destination=master_root / "all-g2a")
        if summaries is None:
            return controlled_dependency_continuation(
                account_id,
                "peer G2A evidence is not yet complete for all five frozen calibration profiles",
            )
        control_dir, control = build_control_k1(
            repo,
            g1a_bundle=g1a_bundle,
            g1a_seal=g1a_seal,
            summaries=summaries,
            master_root=master_root,
            stack=stack,
        )
        print("K1: canonical G2A barrier + scheduler + durability-bound SCIENCE_GO published.", flush=True)
    else:
        acquired = try_acquire_control_worker(repo, g1a_seal=g1a_seal, master_root=master_root)
        if acquired is None:
            return controlled_dependency_continuation(
                account_id,
                "K1 canonical control plane / SCIENCE_GO is not yet published",
            )
        control_dir, control = acquired
        print(f"{account_id}: canonical control plane received and validated.", flush=True)

    stage("SCIENCE_DURABILITY_PREFLIGHT", account_id, guard_token)
    ensure_science_durability(repo, account_id=account_id, username=username, kaggle_env=kaggle_env, control=control)
    assert_clean_science_checkout(repo)

    stage("SCIENTIFIC_QUEUE", account_id, guard_token)
    return run_science(
        repo, account_id=account_id, manifest=manifest, class_map=class_map, image_root=image_root,
        g1a_bundle=g1a_bundle, control_dir=control_dir, control=control, master_root=master_root,
    )


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MASTER_ACCOUNTS:
        raise SystemExit("usage: master_account_driver_v11.py K1|K2|K3")
    account_id = sys.argv[1]
    guard_token = guard_identity(account_id)
    global_clock = time.monotonic()
    os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"] = repr(global_clock)
    os.environ["CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS"] = repr(12 * 3600.0)
    os.environ["CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS"] = repr(3600.0)

    stop = threading.Event()
    thread = threading.Thread(
        target=_heartbeat,
        args=(stop, account_id, guard_token, global_clock),
        daemon=True,
        name=f"tracka-v11-heartbeat-{account_id}",
    )
    thread.start()
    try:
        return _run_account(account_id, guard_token, global_clock)
    finally:
        stop.set()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TimeoutError as exc:
        print(f"CONTROLLED_DEPENDENCY_CONTINUATION: {exc}", flush=True)
        print("Rerun this SAME master notebook in a fresh Batch session; no scientific identity changes.", flush=True)
        raise SystemExit(2)
    except Exception:
        traceback.print_exc()
        raise
