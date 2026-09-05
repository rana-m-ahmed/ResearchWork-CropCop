from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .environment import capture_environment
from .hashing import sha256_json

REQUIRED_RUN_FIELDS = {
    "run_id", "experiment_id", "authority_id", "source_git_commit",
    "config_sha256", "manifest_sha256", "class_map_sha256", "seed",
    "allowed_surfaces", "status",
}
ALLOWED_STATUSES = {
    "PLANNED", "LAUNCHED", "PASS", "FAIL", "INCONCLUSIVE",
    "INTERRUPTED", "NOT_RUN_BY_PREDECLARED_GATE",
}
IDENTITY_FIELDS = (
    "experiment_id", "authority_id", "source_git_commit", "config_sha256",
    "ctc_v2_sha256", "manifest_sha256", "class_map_sha256", "seed",
    "student_init_sha256", "pretrained_sha256", "teacher_sha256",
    "teacher_factory_sha256", "software_stack_sha256", "lane_id",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_summary() -> dict[str, Any]:
    return capture_environment()


def validate_run_record(record: dict) -> None:
    missing = REQUIRED_RUN_FIELDS.difference(record)
    if missing:
        raise ValueError(f"run record missing required fields: {sorted(missing)}")
    if record["status"] not in ALLOWED_STATUSES:
        raise ValueError(f"invalid run status: {record['status']}")
    if "DS-V1-TEST-CONSUMED" in record.get("allowed_surfaces", []):
        raise ValueError("training run cannot authorize V1 test")
    if any(str(x).startswith("DS-EXT-") and str(x).endswith("-SEALED") for x in record.get("allowed_surfaces", [])):
        raise ValueError("training run cannot authorize sealed external surfaces")


def write_run_record(path: str | Path, record: dict) -> str:
    record = dict(record)
    record["updated_at_utc"] = utc_now()
    validate_run_record(record)
    atomic_write_json(path, record)
    return sha256_json(record)


def assert_resume_identity(saved: dict, current: dict) -> None:
    mismatches = {
        f: {"saved": saved.get(f), "current": current.get(f)}
        for f in IDENTITY_FIELDS
        if saved.get(f) != current.get(f)
    }
    if mismatches:
        raise ValueError(f"resume identity mismatch: {json.dumps(mismatches, sort_keys=True)}")


def claim_run_directory(path: str | Path, *, run_id: str, experiment_id: str, lane_id: str) -> dict[str, Any]:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    owner_path = root / ".run_owner.json"
    owner = {"run_id": run_id, "experiment_id": experiment_id, "lane_id": lane_id}
    if owner_path.exists():
        existing = json.loads(owner_path.read_text(encoding="utf-8"))
        if existing != owner:
            raise RuntimeError(f"output-directory collision: existing={existing}, requested={owner}")
        return existing
    fd = os.open(str(owner_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = (json.dumps(owner, indent=2, sort_keys=True) + "\n").encode("utf-8")
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return owner
