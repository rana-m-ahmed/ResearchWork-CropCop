from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, fsync_directory


class CheckpointError(RuntimeError):
    pass


class CheckpointCorruptionError(CheckpointError):
    pass


@dataclass(frozen=True)
class CheckpointRef:
    relative_path: str
    sha256: str
    bytes: int
    generation: int
    optimizer_step: int
    epoch: int
    batch_in_epoch: int
    identity_sha256: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_torch(path: Path) -> dict[str, Any]:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointCorruptionError("checkpoint payload is not a mapping")
    return payload


def _validate_payload(payload: dict[str, Any], *, expected_identity: dict[str, Any]) -> None:
    if payload.get("identity") != expected_identity:
        raise CheckpointCorruptionError("checkpoint scientific identity mismatch")
    required = {"student", "optimizer", "scheduler", "scaler", "rng", "epoch", "batch_in_epoch", "optimizer_step", "selection_state"}
    missing = required.difference(payload)
    if missing:
        raise CheckpointCorruptionError(f"checkpoint missing fields: {sorted(missing)}")


def _index_path(root: Path) -> Path:
    return root / "checkpoint_index.json"


def read_index(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = _index_path(root)
    if not path.exists():
        return {"schema_version": "2.0", "generation": 0, "latest": None, "previous": None, "selected": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "2.0":
        raise CheckpointError(f"unsupported checkpoint index schema: {payload.get('schema_version')}")
    return payload


def _verify_ref(root: Path, ref: dict[str, Any], *, expected_identity: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = root / ref["relative_path"]
    if not path.exists():
        raise CheckpointCorruptionError(f"checkpoint file missing: {path}")
    if path.stat().st_size != int(ref["bytes"]):
        raise CheckpointCorruptionError(f"checkpoint byte-size mismatch: {path}")
    digest = _sha256(path)
    if digest != ref["sha256"]:
        raise CheckpointCorruptionError(f"checkpoint SHA-256 mismatch: {path}")
    payload = _load_torch(path)
    _validate_payload(payload, expected_identity=expected_identity)
    return path, payload


def save_torch_checkpoint(
    root: str | Path,
    *,
    kind: str,
    payload: dict[str, Any],
    expected_identity: dict[str, Any],
    min_free_bytes: int = 0,
) -> tuple[CheckpointRef, float]:
    if kind not in {"latest", "selected"}:
        raise ValueError("kind must be latest or selected")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if min_free_bytes > 0:
        free = shutil.disk_usage(root).free
        if free < min_free_bytes:
            raise OSError(f"insufficient free disk for crash-safe checkpoint: free={free}, required={min_free_bytes}")
    objects = root / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    index = read_index(root)
    generation = int(index.get("generation", 0)) + 1
    started = time.perf_counter()

    import torch
    fd, tmp_name = tempfile.mkstemp(prefix=f".{kind}.{generation}.", suffix=".ckpt.tmp", dir=objects)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        with tmp.open("rb+") as f:
            os.fsync(f.fileno())
        digest = _sha256(tmp)
        loaded = _load_torch(tmp)
        _validate_payload(loaded, expected_identity=expected_identity)
        final = objects / f"{kind}.g{generation:08d}.{digest[:16]}.ckpt"
        os.replace(tmp, final)
        fsync_directory(objects)
        ref = CheckpointRef(
            relative_path=final.relative_to(root).as_posix(),
            sha256=digest,
            bytes=final.stat().st_size,
            generation=generation,
            optimizer_step=int(payload["optimizer_step"]),
            epoch=int(payload["epoch"]),
            batch_in_epoch=int(payload["batch_in_epoch"]),
            identity_sha256=str(payload["identity_sha256"]),
            created_at_utc=str(payload["created_at_utc"]),
        )
        new_index = dict(index)
        new_index["generation"] = generation
        if kind == "latest":
            new_index["previous"] = index.get("latest")
            new_index["latest"] = ref.to_dict()
        else:
            new_index["selected"] = ref.to_dict()
        atomic_write_json(_index_path(root), new_index)
        _prune_unreferenced(root, new_index)
        return ref, time.perf_counter() - started
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _prune_unreferenced(root: Path, index: dict[str, Any]) -> None:
    keep = {r["relative_path"] for r in (index.get("latest"), index.get("previous"), index.get("selected")) if r}
    objects = root / "objects"
    if not objects.exists():
        return
    for path in objects.glob("*.ckpt"):
        rel = path.relative_to(root).as_posix()
        if rel not in keep:
            try:
                path.unlink()
            except OSError:
                pass


def recover_latest(root: str | Path, *, expected_identity: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    root = Path(root)
    index = read_index(root)
    errors = []
    candidates = []
    seen = set()
    for label in ("latest", "selected", "previous"):
        ref = index.get(label)
        if ref and ref.get("relative_path") not in seen:
            seen.add(ref.get("relative_path"))
            candidates.append((label, ref))
    candidates.sort(key=lambda item: int(item[1].get("generation", 0)), reverse=True)
    for label, ref in candidates:
        try:
            path, payload = _verify_ref(root, ref, expected_identity=expected_identity)
            event = {
                "event": "RECOVERY_CHECKPOINT_SELECTED",
                "candidate": label,
                "recovered": ref,
                "prior_candidate_errors": errors,
            }
            return path, payload, event
        except Exception as exc:
            errors.append({"candidate": label, "error": f"{type(exc).__name__}: {exc}"})
    raise CheckpointCorruptionError(f"no valid recovery checkpoint; attempts={errors}")


def verify_selected(root: str | Path, *, expected_identity: dict[str, Any], expected_sha256: str | None = None) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    index = read_index(root)
    ref = index.get("selected")
    if not ref:
        raise CheckpointCorruptionError("selected checkpoint is not indexed")
    if expected_sha256 and ref.get("sha256") != expected_sha256:
        raise CheckpointCorruptionError("selected checkpoint index SHA differs from selection state")
    return _verify_ref(root, ref, expected_identity=expected_identity)


def export_recovery_bundle(root: str | Path, destination: str | Path) -> dict[str, Any]:
    root = Path(root)
    destination = Path(destination)
    index = read_index(root)
    if not any(index.get(k) for k in ("latest", "previous", "selected")):
        raise CheckpointError(f"no verified checkpoint refs available for durable export: {root}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for ref in (index.get("latest"), index.get("previous"), index.get("selected")):
        if not ref:
            continue
        src = root / ref["relative_path"]
        dst = destination / ref["relative_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if _sha256(dst) != ref["sha256"]:
            raise CheckpointCorruptionError(f"durable-copy verification failed: {dst}")
        copied.append(ref)
    atomic_write_json(destination / "checkpoint_index.json", index)
    return {"index": index, "copied": copied}
