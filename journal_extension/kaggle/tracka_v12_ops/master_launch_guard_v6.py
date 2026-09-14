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

STATUS_SCHEMA = "1.0"
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
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _status_matches_recent_owner(status: dict | None, *, account_id: str, runtime_sha: str, wait_started_unix: float) -> bool:
    if not status:
        return False
    return (
        status.get("schema_version") == STATUS_SCHEMA
        and status.get("account_id") == account_id
        and status.get("science_sha") == SCIENCE_SHA
        and status.get("operator_runtime_sha") == runtime_sha
        and status.get("state") == "FINISHED"
        and isinstance(status.get("return_code"), int)
        and float(status.get("finished_unix", 0.0)) >= wait_started_unix - 1.0
    )


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
        "started_unix": started,
    })
    print(f"MASTER_LAUNCH_GUARD OWNER account={account_id} pid={os.getpid()} launch={launch_id[:12]}", flush=True)

    driver = Path(__file__).resolve().parent / "master_account_driver_v6.py"
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
        "started_unix": started,
        "finished_unix": time.time(),
        "return_code": rc,
    })
    print(f"MASTER_LAUNCH_GUARD OWNER_FINISHED account={account_id} rc={rc} launch={launch_id[:12]}", flush=True)
    return rc


def launch_or_follow(account_id: str) -> int:
    if account_id not in MASTER_ACCOUNTS:
        raise OperatorError(f"unknown master account: {account_id}")

    runtime_sha = operator_runtime_head()
    guard_root = Path("/kaggle/working/.cropcop_tracka_master_guard")
    guard_root.mkdir(parents=True, exist_ok=True)
    lock_path = guard_root / f"{account_id}.lock"
    status_path = guard_root / f"{account_id}.status.json"
    wait_started_unix = time.time()
    wait_deadline = time.monotonic() + MAX_WAIT_SECONDS
    observed_busy = False

    with lock_path.open("a+") as lock_handle:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if observed_busy:
                    status = _load_status(status_path)
                    if _status_matches_recent_owner(
                        status,
                        account_id=account_id,
                        runtime_sha=runtime_sha,
                        wait_started_unix=wait_started_unix,
                    ):
                        rc = int(status["return_code"])
                        print(
                            f"MASTER_LAUNCH_GUARD FOLLOWER_RESULT account={account_id} rc={rc} "
                            f"owner_launch={str(status.get('launch_id', ''))[:12]}",
                            flush=True,
                        )
                        return rc
                    print(
                        "MASTER_LAUNCH_GUARD previous owner released without a valid matching terminal status; "
                        "this follower is taking ownership safely.",
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
                        f"MASTER_LAUNCH_GUARD DUPLICATE_DETECTED account={account_id}; "
                        "waiting for canonical owner instead of starting a second master.",
                        flush=True,
                    )
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError(f"duplicate master owner did not finish within {MAX_WAIT_SECONDS}s")
                time.sleep(POLL_SECONDS)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: master_launch_guard_v6.py K1|K2|K3")
    return launch_or_follow(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
