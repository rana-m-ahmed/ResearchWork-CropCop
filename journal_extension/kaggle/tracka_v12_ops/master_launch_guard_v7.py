from __future__ import annotations

import fcntl
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import MASTER_ACCOUNTS, SCIENCE_SHA, OperatorError, operator_runtime_head

STATUS_SCHEMA = "2.0"
POLL_SECONDS = 2.0
MAX_WAIT_SECONDS = 12 * 3600.0


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperatorError(f"master launch status is unreadable: {path}: {exc}") from exc
    return value if isinstance(value, dict) else None


def _matching_terminal_code(status: dict | None, *, account_id: str, runtime_sha: str) -> int | None:
    if not status:
        return None
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("account_id") != account_id
        or status.get("science_sha") != SCIENCE_SHA
        or status.get("operator_runtime_sha") != runtime_sha
    ):
        return None
    if status.get("state") != "FINISHED" or not isinstance(status.get("return_code"), int):
        return None
    return int(status["return_code"])


def _run_owner(account_id: str, *, lock_path: Path, status_path: Path, runtime_sha: str) -> int:
    launch_id = secrets.token_hex(16)
    started = time.time()
    os.environ["CROPCOP_MASTER_GUARD_TOKEN"] = launch_id
    os.environ["CROPCOP_MASTER_GUARD_ACCOUNT"] = account_id
    os.environ["CROPCOP_MASTER_GUARD_LOCK"] = str(lock_path)

    _atomic_json(status_path, {
        "schema_version": STATUS_SCHEMA,
        "state": "RUNNING",
        "account_id": account_id,
        "science_sha": SCIENCE_SHA,
        "operator_runtime_sha": runtime_sha,
        "launch_id": launch_id,
        "owner_pid": os.getpid(),
        "owner_ppid": os.getppid(),
        "started_unix": started,
    })
    print(
        f"MASTER_LAUNCH_GUARD_V7 OWNER account={account_id} pid={os.getpid()} "
        f"ppid={os.getppid()} launch={launch_id[:12]}",
        flush=True,
    )

    driver = Path(__file__).resolve().parent / "master_account_driver_v7.py"
    try:
        cp = subprocess.run([sys.executable, "-u", str(driver), account_id], cwd=driver.parent)
        rc = int(cp.returncode)
    except BaseException as exc:
        rc = 1
        _atomic_json(status_path, {
            "schema_version": STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": account_id,
            "science_sha": SCIENCE_SHA,
            "operator_runtime_sha": runtime_sha,
            "launch_id": launch_id,
            "owner_pid": os.getpid(),
            "owner_ppid": os.getppid(),
            "started_unix": started,
            "finished_unix": time.time(),
            "return_code": rc,
            "launcher_exception": f"{type(exc).__name__}: {exc}",
        })
        raise

    _atomic_json(status_path, {
        "schema_version": STATUS_SCHEMA,
        "state": "FINISHED",
        "account_id": account_id,
        "science_sha": SCIENCE_SHA,
        "operator_runtime_sha": runtime_sha,
        "launch_id": launch_id,
        "owner_pid": os.getpid(),
        "owner_ppid": os.getppid(),
        "started_unix": started,
        "finished_unix": time.time(),
        "return_code": rc,
    })
    print(
        f"MASTER_LAUNCH_GUARD_V7 OWNER_FINISHED account={account_id} rc={rc} launch={launch_id[:12]}",
        flush=True,
    )
    return rc


def launch_or_follow(account_id: str) -> int:
    if account_id not in MASTER_ACCOUNTS:
        raise OperatorError(f"unknown master account: {account_id}")

    runtime_sha = operator_runtime_head()
    guard_root = Path("/kaggle/working/.cropcop_tracka_master_guard_v7")
    guard_root.mkdir(parents=True, exist_ok=True)
    lock_path = guard_root / f"{account_id}.lock"
    status_path = guard_root / f"{account_id}.status.json"
    wait_deadline = time.monotonic() + MAX_WAIT_SECONDS
    observed_busy = False

    with lock_path.open("a+") as lock_handle:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                status = _load_status(status_path)
                prior_code = _matching_terminal_code(status, account_id=account_id, runtime_sha=runtime_sha)
                if prior_code is not None:
                    print(
                        f"MASTER_LAUNCH_GUARD_V7 TERMINAL_REUSE account={account_id} rc={prior_code} "
                        f"owner_launch={str(status.get('launch_id', ''))[:12]}; refusing duplicate rerun.",
                        flush=True,
                    )
                    return prior_code

                if status and status.get("state") == "RUNNING":
                    raise OperatorError(
                        "stale RUNNING master marker found after lock acquisition; use a fresh Kaggle Batch session "
                        "instead of risking duplicate/recovered execution in-place"
                    )

                if observed_busy:
                    print(
                        "MASTER_LAUNCH_GUARD_V7 prior owner released without a matching terminal marker; "
                        "taking ownership only because no canonical terminal result exists.",
                        flush=True,
                    )
                return _run_owner(
                    account_id,
                    lock_path=lock_path,
                    status_path=status_path,
                    runtime_sha=runtime_sha,
                )
            except BlockingIOError:
                if not observed_busy:
                    observed_busy = True
                    print(
                        f"MASTER_LAUNCH_GUARD_V7 DUPLICATE_DETECTED account={account_id}; "
                        "waiting for canonical owner instead of starting a second master.",
                        flush=True,
                    )
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(f"duplicate master owner did not finish within {MAX_WAIT_SECONDS}s")
                time.sleep(POLL_SECONDS)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: master_launch_guard_v7.py K1|K2|K3")
    return launch_or_follow(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
