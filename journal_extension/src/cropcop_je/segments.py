from __future__ import annotations

import json
import os
import platform
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_segment_id(run_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{run_id}-SEG-{stamp}-{uuid.uuid4().hex[:8]}"


def host_identity() -> dict[str, Any]:
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "pid": os.getpid()}


def append_segment_event(path: str | Path, event: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n"
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
