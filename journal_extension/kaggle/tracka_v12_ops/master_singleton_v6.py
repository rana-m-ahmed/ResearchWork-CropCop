from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import OperatorError, operator_runtime_head


@dataclass
class MasterExecutionLease:
    account_id: str
    lock_path: Path
    marker_path: Path
    fd: int | None
    mirror_code: int | None = None

    @property
    def is_primary(self) -> bool:
        return self.fd is not None and self.mirror_code is None

    def finish(self, code: int, detail: str = "") -> None:
        if not self.is_primary:
            return
        status = "PASS" if code == 0 else "CONTROLLED_CONTINUATION" if code == 2 else "FAILED"
        _atomic_marker(
            self.marker_path,
            {
                "schema_version": "1.0",
                "account_id": self.account_id,
                "runtime_sha": operator_runtime_head(),
                "pid": os.getpid(),
                "status": status,
                "exit_code": int(code),
                "detail": detail,
                "finished_unix": time.time(),
            },
        )
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def _root() -> Path:
    root = Path(os.environ.get("CROPCOP_MASTER_SINGLETON_ROOT", "/kaggle/working"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_marker(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OperatorError(f"master singleton marker is unreadable: {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else None


def _atomic_marker(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _terminal_code(marker: dict | None, *, account_id: str) -> int | None:
    if not marker:
        return None
    if marker.get("account_id") != account_id:
        raise OperatorError("master singleton marker account mismatch")
    status = marker.get("status")
    if status in {"PASS", "CONTROLLED_CONTINUATION", "FAILED"}:
        return int(marker.get("exit_code", 1))
    return None


def acquire_master_execution(account_id: str, *, wait_seconds: float = 12 * 3600.0) -> MasterExecutionLease:
    root = _root()
    lock_path = root / f".cropcop-tracka-master-{account_id}.lock"
    marker_path = root / f".cropcop-tracka-master-{account_id}.json"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        print(
            f"DUPLICATE_MASTER_INVOCATION {account_id}: another driver owns the session lock; "
            "waiting for its terminal marker instead of entering the critical section.",
            flush=True,
        )
        deadline = time.monotonic() + float(wait_seconds)
        while time.monotonic() < deadline:
            marker = _read_marker(marker_path)
            code = _terminal_code(marker, account_id=account_id)
            if code is not None:
                print(
                    f"DUPLICATE_MASTER_MIRROR {account_id}: primary finished with rc={code}; "
                    "this duplicate invocation will mirror that result without rerunning stages.",
                    flush=True,
                )
                return MasterExecutionLease(account_id, lock_path, marker_path, None, mirror_code=code)
            time.sleep(2.0)
        raise TimeoutError(f"duplicate {account_id} master invocation did not reach a terminal marker")

    marker = _read_marker(marker_path)
    previous_code = _terminal_code(marker, account_id=account_id)
    if previous_code is not None:
        print(
            f"MASTER_SESSION_ALREADY_TERMINAL {account_id}: rc={previous_code}; "
            "refusing to rerun the same account in the same /kaggle/working session.",
            flush=True,
        )
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return MasterExecutionLease(account_id, lock_path, marker_path, None, mirror_code=previous_code)

    if marker and marker.get("status") == "RUNNING":
        raise OperatorError(
            "stale RUNNING master marker found after lock acquisition; use a fresh Kaggle Batch session "
            "instead of risking duplicate/recovered execution in-place"
        )

    _atomic_marker(
        marker_path,
        {
            "schema_version": "1.0",
            "account_id": account_id,
            "runtime_sha": operator_runtime_head(),
            "pid": os.getpid(),
            "status": "RUNNING",
            "started_unix": time.time(),
        },
    )
    print(
        f"MASTER_SINGLETON_ACQUIRED {account_id}: pid={os.getpid()} lock={lock_path}",
        flush=True,
    )
    return MasterExecutionLease(account_id, lock_path, marker_path, fd)
