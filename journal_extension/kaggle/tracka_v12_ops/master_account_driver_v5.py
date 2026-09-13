from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    MASTER_ACCOUNTS,
    OperatorError,
    assert_clean_science_checkout,
    assert_kaggle_batch,
    assert_kaggle_paths,
    assert_python_version,
    ensure_locked_stack,
    ensure_science_checkout,
    github_write_preflight,
    load_github_token,
    load_kaggle_credentials,
    operator_runtime_head,
)
from master_preflight import resolve_master_inputs
from master_g1a import acquire_canonical_g1a_worker, ensure_canonical_g1a_k1
from master_g2a import collect_all_g2a, ensure_account_g2a
from master_control import acquire_control_worker, build_control_k1
from master_publication_v4 import install_stage_publication_hooks
from master_science_v4 import ensure_science_durability, run_science


def stage(name: str, account_id: str) -> None:
    print(f"\n=== TRACKA_V12_MASTER {account_id} :: {name} ===", flush=True)


def gpu_names() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def assert_t4x2() -> list[str]:
    names = gpu_names()
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
        raise OperatorError(f"master notebook requires Kaggle T4 x2; observed {names}")
    return names


def assert_expected_account(account_id: str, username: str) -> None:
    expected = str(os.environ.get("CROPCOP_EXPECTED_KAGGLE_USERNAME", "") or "").strip()
    if expected and username.casefold() != expected.casefold():
        raise OperatorError(
            f"{account_id} Kaggle account mismatch: expected {expected!r}, credential username is {username!r}. "
            "Fix the Kaggle secrets or remove the optional expectation before continuing."
        )
    print(f"KAGGLE_ACCOUNT_BINDING {account_id}={username}", flush=True)


def retry_operator_call(label: str, fn, *, attempts: int = 3):
    """Retry deterministic infrastructure I/O only, before scientific execution."""
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
            print(f"{label} attempt {attempt}/{attempts} failed: {type(exc).__name__}; retrying in {delay}s")
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MASTER_ACCOUNTS:
        raise SystemExit("usage: master_account_driver_v5.py K1|K2|K3")
    account_id = sys.argv[1]
    global_clock = time.monotonic()

    stage("RUNTIME_AND_HARDWARE_PREFLIGHT", account_id)
    print("Pinned Track-A master operator runtime:", operator_runtime_head())
    assert_kaggle_paths()
    assert_kaggle_batch()
    assert_python_version()
    print("GPU inventory:", assert_t4x2())

    username, key = load_kaggle_credentials()
    assert_expected_account(account_id, username)
    load_github_token()
    kaggle_env = dict(os.environ)
    kaggle_env["KAGGLE_USERNAME"] = username
    kaggle_env["KAGGLE_KEY"] = key

    master_root = Path(f"/kaggle/working/TRACKA_V12_MASTER_{account_id}")
    master_root.mkdir(parents=True, exist_ok=True)
    os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"] = repr(global_clock)
    os.environ["CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS"] = repr(12 * 3600.0)
    os.environ["CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS"] = repr(3600.0)

    stage("FROZEN_INPUT_RESOLUTION", account_id)
    manifest, class_map, image_root = resolve_master_inputs(account_id)
    print("BOUND_MANIFEST=", manifest)
    print("BOUND_CLASS_MAP=", class_map)
    print("BOUND_IMAGE_ROOT=", image_root)
    if account_id == "K1":
        print("BOUND_PRINCIPAL_G1=", os.environ.get("CROPCOP_PRINCIPAL_G1", ""))

    stage("SCIENCE_SOURCE_AND_GITHUB_PREFLIGHT", account_id)
    repo = retry_operator_call("Frozen science checkout", ensure_science_checkout)
    assert_clean_science_checkout(repo)
    retry_operator_call("GitHub evidence write preflight", lambda: github_write_preflight(repo))

    stage("EXACT_EXECUTION_STACK", account_id)
    stack = ensure_locked_stack(repo)
    assert_clean_science_checkout(repo)
    print("DEPENDENCY_LOCK_SHA256=", stack["dependency_lock_sha256"])

    # G1A/G2A/control evidence is parent-only and recovery-safe from here onward.
    install_stage_publication_hooks()

    stage("CANONICAL_G1A", account_id)
    if account_id == "K1":
        g1a_bundle, g1a_seal, shared_locator = ensure_canonical_g1a_k1(
            repo,
            manifest=manifest,
            class_map=class_map,
            image_root=image_root,
            username=username,
            kaggle_env=kaggle_env,
            stack=stack,
            master_root=master_root,
        )
    else:
        g1a_bundle, g1a_seal, shared_locator = acquire_canonical_g1a_worker(
            repo,
            kaggle_env=kaggle_env,
            master_root=master_root,
        )
    print(f"{account_id}: canonical G1A PASS {g1a_seal['g1a_seal_sha256']} via {shared_locator}")

    stage("ACCOUNT_G2A", account_id)
    ensure_account_g2a(
        repo,
        account_id=account_id,
        username=username,
        kaggle_env=kaggle_env,
        manifest=manifest,
        class_map=class_map,
        image_root=image_root,
        g1a_bundle=g1a_bundle,
        g1a_seal=g1a_seal,
        master_root=master_root,
        global_clock=global_clock,
    )
    print(f"{account_id}: account G2A evidence canonicalized.")

    stage("CONTROL_PLANE", account_id)
    if account_id == "K1":
        summaries = collect_all_g2a(repo, g1a_seal=g1a_seal, destination=master_root / "all-g2a")
        control_dir, control = build_control_k1(
            repo,
            g1a_bundle=g1a_bundle,
            g1a_seal=g1a_seal,
            summaries=summaries,
            master_root=master_root,
            stack=stack,
        )
        print("K1: canonical G2A barrier + scheduler + durability-bound SCIENCE_GO published.")
    else:
        control_dir, control = acquire_control_worker(repo, g1a_seal=g1a_seal, master_root=master_root)
        print(f"{account_id}: canonical control plane received and validated.")

    stage("SCIENCE_DURABILITY_PREFLIGHT", account_id)
    ensure_science_durability(
        repo,
        account_id=account_id,
        username=username,
        kaggle_env=kaggle_env,
        control=control,
    )
    assert_clean_science_checkout(repo)

    stage("SCIENTIFIC_QUEUE", account_id)
    return run_science(
        repo,
        account_id=account_id,
        manifest=manifest,
        class_map=class_map,
        image_root=image_root,
        g1a_bundle=g1a_bundle,
        control_dir=control_dir,
        control=control,
        master_root=master_root,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TimeoutError as exc:
        print(f"DEPENDENCY_WAIT_TIMEOUT: {exc}")
        print("Rerun this SAME master notebook in a fresh Batch session; no scientific identity changes.")
        raise SystemExit(2)
    except Exception:
        traceback.print_exc()
        raise
