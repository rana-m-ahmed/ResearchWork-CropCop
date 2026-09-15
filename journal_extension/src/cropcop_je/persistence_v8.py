from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .checkpointing import export_recovery_bundle
from .g1_publication import G1PublicationError, preflight_private_target
from .hashing import sha256_file
from .persistence import (
    DurableStore,
    FilesystemStore,
    KagglePrivateDatasetStore,
    PersistenceError,
    SyncResult,
)


@dataclass
class GenerationAwareSyncResult(SyncResult):
    previous_version_number: int | None = None
    confirmed_version_number: int | None = None
    generation_marker_sha256: str | None = None
    generation_roundtrip_verified: bool = False


class GenerationNotCurrentError(PersistenceError):
    """The downloaded Kaggle archive is validly readable but not the requested generation yet."""


def _version_number(payload: dict[str, Any]) -> int:
    value = payload.get("current_version_number")
    if value is None:
        raise PersistenceError("Kaggle durability state has no current_version_number")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid Kaggle durability version number: {value!r}") from exc


def _ready_state(dataset_slug: str) -> dict[str, Any]:
    try:
        return preflight_private_target(dataset_slug)
    except G1PublicationError as exc:
        raise PersistenceError(str(exc)) from exc


class GenerationAwareKagglePrivateDatasetStore(KagglePrivateDatasetStore):
    """Kaggle private durability with explicit generation settlement and round-trip identity."""

    def _wait_for_new_generation(
        self,
        previous_version: int,
        *,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        last_state: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                state = _ready_state(self.dataset_slug)
                last_state = state
                if _version_number(state) > previous_version:
                    return state
            except PersistenceError as exc:
                last_error = exc
            time.sleep(poll_interval_seconds)
        detail = f"last_state={last_state}" if last_state is not None else f"last_error={last_error}"
        raise PersistenceError(
            "Kaggle durable dataset did not expose a newer ready generation before timeout; " + detail
        )

    @staticmethod
    def _validate_generation_payload(
        root: Path,
        *,
        run_id: str,
        segment_id: str | None = None,
        expected_nonce: str | None = None,
    ) -> dict[str, Any]:
        marker_path = root / "durable_sync.json"
        index_path = root / "checkpoint_index.json"
        if not marker_path.is_file() or not index_path.is_file():
            raise GenerationNotCurrentError("Kaggle durable generation is missing marker or checkpoint index")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PersistenceError("Kaggle durable generation marker is not valid JSON") from exc
        if marker.get("schema_version") != "2.0" or marker.get("complete") is not True:
            raise PersistenceError("Kaggle durable generation marker is incomplete or unsupported")
        if marker.get("run_id") != run_id:
            raise GenerationNotCurrentError("Kaggle durable generation run identity mismatch")
        if segment_id is not None and marker.get("segment_id") != segment_id:
            raise GenerationNotCurrentError("Kaggle durable generation segment identity mismatch")
        if expected_nonce is not None and marker.get("sync_nonce") != expected_nonce:
            raise GenerationNotCurrentError("Kaggle durable generation nonce mismatch")
        if marker.get("checkpoint_index_sha256") != sha256_file(index_path):
            raise PersistenceError("Kaggle durable generation checkpoint-index hash mismatch")
        return marker

    def _download_latest(self, destination: Path) -> None:
        self._run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                self.dataset_slug,
                "-p",
                str(destination),
                "--unzip",
                "-q",
            ]
        )

    def _wait_for_exact_downloadable_generation(
        self,
        destination: Path,
        *,
        run_id: str,
        segment_id: str | None = None,
        expected_nonce: str | None = None,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """Wait until Kaggle serves the exact requested generation, not merely newer metadata."""
        deadline = time.monotonic() + timeout_seconds
        last_transient: Exception | None = None
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=False)
            try:
                self._download_latest(destination)
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                last_transient = exc
            else:
                try:
                    return self._validate_generation_payload(
                        destination,
                        run_id=run_id,
                        segment_id=segment_id,
                        expected_nonce=expected_nonce,
                    )
                except GenerationNotCurrentError as exc:
                    last_transient = exc
                except PersistenceError:
                    raise
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_interval_seconds)
        raise PersistenceError(
            "Kaggle durable generation metadata advanced but the exact generation never became "
            f"downloadable/valid before timeout after {attempts} attempts; last_transient={last_transient}"
        )

    def sync(self, source_dir: Path, *, run_id: str, segment_id: str) -> SyncResult:
        before = _ready_state(self.dataset_slug)
        previous_version = _version_number(before)
        nonce = uuid.uuid4().hex

        root_ctx = tempfile.TemporaryDirectory(dir=self.work_root) if self.work_root else tempfile.TemporaryDirectory()
        with root_ctx as td:
            staging = Path(td)
            export_recovery_bundle(source_dir, staging)
            marker = {
                "schema_version": "2.0",
                "run_id": run_id,
                "segment_id": segment_id,
                "sync_nonce": nonce,
                "checkpoint_index_sha256": sha256_file(staging / "checkpoint_index.json"),
                "previous_version_number": previous_version,
                "complete": True,
            }
            atomic_write_json(staging / "durable_sync.json", marker)
            metadata = {
                "title": self.dataset_slug.split("/", 1)[1],
                "id": self.dataset_slug,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            }
            atomic_write_json(staging / "dataset-metadata.json", metadata)
            message = f"CropCop recovery {run_id} {segment_id}"
            self._run(
                [
                    "kaggle",
                    "datasets",
                    "version",
                    "-p",
                    str(staging),
                    "-m",
                    message,
                    "-q",
                    "-r",
                    "zip",
                ]
            )
            files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())

        settled = self._wait_for_new_generation(previous_version)
        confirmed_version = _version_number(settled)
        verify_ctx = tempfile.TemporaryDirectory(dir=self.work_root) if self.work_root else tempfile.TemporaryDirectory()
        with verify_ctx as td:
            verify_root = Path(td)
            self._wait_for_exact_downloadable_generation(
                verify_root,
                run_id=run_id,
                segment_id=segment_id,
                expected_nonce=nonce,
            )
            marker_sha = sha256_file(verify_root / "durable_sync.json")

        return GenerationAwareSyncResult(
            backend="kaggle_private_dataset",
            locator=self.dataset_slug,
            files=files,
            previous_version_number=previous_version,
            confirmed_version_number=confirmed_version,
            generation_marker_sha256=marker_sha,
            generation_roundtrip_verified=True,
        )

    def restore(self, destination_dir: Path, *, run_id: str) -> bool:
        _ready_state(self.dataset_slug)
        root_ctx = tempfile.TemporaryDirectory(dir=self.work_root) if self.work_root else tempfile.TemporaryDirectory()
        with root_ctx as td:
            staging = Path(td)
            self._wait_for_exact_downloadable_generation(staging, run_id=run_id)
            destination_dir.mkdir(parents=True, exist_ok=True)
            for path in staging.rglob("*"):
                if path.is_file() and path.name not in {"dataset-metadata.json", "durable_sync.json"}:
                    rel = path.relative_to(staging)
                    dst = destination_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)
            return True


def build_store_v8(kind: str, locator: str) -> DurableStore:
    if kind == "filesystem":
        return FilesystemStore(locator)
    if kind == "kaggle-dataset":
        return GenerationAwareKagglePrivateDatasetStore(locator)
    raise ValueError(f"unsupported durable store kind: {kind}")
