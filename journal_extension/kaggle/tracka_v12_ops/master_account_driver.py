from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    MASTER_ACCOUNTS, OperatorError, assert_clean_science_checkout, assert_kaggle_batch,
    assert_kaggle_paths, assert_python_version, ensure_locked_stack, ensure_science_checkout,
    github_write_preflight, load_github_token, load_kaggle_credentials, operator_runtime_head,
)
from master_preflight import resolve_master_inputs
from master_g1a import acquire_canonical_g1a_worker, ensure_canonical_g1a_k1
from master_g2a import collect_all_g2a, ensure_account_g2a
from master_control import acquire_control_worker, build_control_k1
from master_science import ensure_science_durability, run_science


def gpu_names() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def assert_t4x2() -> list[str]:
    names = gpu_names()
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
        raise OperatorError(f"master notebook requires Kaggle T4 x2; observed {names}")
    return names


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in MASTER_ACCOUNTS:
        raise SystemExit("usage: master_account_driver.py K1|K2|K3")
    account_id = sys.argv[1]
    global_clock = time.monotonic()
    print("Pinned Track-A master operator runtime:", operator_runtime_head())
    assert_kaggle_paths()
    assert_kaggle_batch()
    assert_python_version()
    assert_t4x2()

    username, key = load_kaggle_credentials()
    load_github_token()
    kaggle_env = dict(os.environ)
    kaggle_env["KAGGLE_USERNAME"] = username
    kaggle_env["KAGGLE_KEY"] = key
    master_root = Path(f"/kaggle/working/TRACKA_V12_MASTER_{account_id}")
    master_root.mkdir(parents=True, exist_ok=True)
    os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"] = repr(global_clock)
    os.environ["CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS"] = repr(12 * 3600.0)
    os.environ["CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS"] = repr(3600.0)

    # Fail fast on immutable Kaggle inputs before cloning/installing the heavy frozen stack.
    manifest, class_map, image_root = resolve_master_inputs(account_id)

    repo = ensure_science_checkout()
    stack = ensure_locked_stack(repo)
    assert_clean_science_checkout(repo)
    github_write_preflight(repo)

    if account_id == "K1":
        g1a_bundle, g1a_seal, shared_locator = ensure_canonical_g1a_k1(
            repo, manifest=manifest, class_map=class_map, image_root=image_root,
            username=username, kaggle_env=kaggle_env, stack=stack, master_root=master_root,
        )
    else:
        g1a_bundle, g1a_seal, shared_locator = acquire_canonical_g1a_worker(
            repo, kaggle_env=kaggle_env, master_root=master_root,
        )
    print(f"{account_id}: canonical G1A PASS {g1a_seal['g1a_seal_sha256']} via {shared_locator}")

    ensure_account_g2a(
        repo, account_id=account_id, username=username, kaggle_env=kaggle_env,
        manifest=manifest, class_map=class_map, image_root=image_root,
        g1a_bundle=g1a_bundle, g1a_seal=g1a_seal, master_root=master_root,
        global_clock=global_clock,
    )
    print(f"{account_id}: account G2A evidence canonicalized.")

    if account_id == "K1":
        summaries = collect_all_g2a(repo, g1a_seal=g1a_seal, destination=master_root / "all-g2a")
        control_dir, control = build_control_k1(
            repo, g1a_bundle=g1a_bundle, g1a_seal=g1a_seal, summaries=summaries,
            master_root=master_root, stack=stack,
        )
        print("K1: canonical G2A barrier + scheduler + durability-bound SCIENCE_GO published.")
    else:
        control_dir, control = acquire_control_worker(repo, g1a_seal=g1a_seal, master_root=master_root)
        print(f"{account_id}: canonical control plane received and validated.")

    ensure_science_durability(
        repo, account_id=account_id, username=username, kaggle_env=kaggle_env, control=control,
    )
    assert_clean_science_checkout(repo)
    return run_science(
        repo, account_id=account_id, manifest=manifest, class_map=class_map,
        image_root=image_root, g1a_bundle=g1a_bundle, control_dir=control_dir,
        control=control, master_root=master_root,
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
