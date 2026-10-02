from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, fsync_directory
from .checkpointing import export_recovery_bundle


class PersistenceError(RuntimeError):
    pass


@dataclass
class SyncResult:
    backend: str
    locator: str
    files: list[str]
    status: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class DurableStore:
    def sync(self, source_dir: Path, *, run_id: str, segment_id: str) -> SyncResult:
        raise NotImplementedError

    def restore(self, destination_dir: Path, *, run_id: str) -> bool:
        raise NotImplementedError


class FilesystemStore(DurableStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def _previous_dir(self, run_id: str) -> Path:
        return self.root / f".{run_id}.previous"

    @staticmethod
    def _valid_generation(path: Path, run_id: str) -> bool:
        marker = path / "durable_sync.json"
        if not marker.exists():
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("complete") is True and payload.get("run_id") == run_id and (path / "checkpoint_index.json").exists()

    def sync(self, source_dir: Path, *, run_id: str, segment_id: str) -> SyncResult:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._run_dir(run_id)
        staging = self.root / f".{run_id}.{segment_id}.staging"
        old = self._previous_dir(run_id)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        export_recovery_bundle(source_dir, staging)
        marker = {"schema_version": "1.0", "run_id": run_id, "segment_id": segment_id, "complete": True}
        atomic_write_json(staging / "durable_sync.json", marker)
        fsync_directory(staging)

        if old.exists():
            shutil.rmtree(old)
        if target.exists():
            os.replace(target, old)
            fsync_directory(self.root)
        os.replace(staging, target)
        fsync_directory(self.root)
        if not self._valid_generation(target, run_id):
            raise PersistenceError("new filesystem durable generation failed post-commit validation")
        if old.exists():
            shutil.rmtree(old)
            fsync_directory(self.root)
        files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
        return SyncResult("filesystem", str(target), files)

    def restore(self, destination_dir: Path, *, run_id: str) -> bool:
        active = self._run_dir(run_id)
        previous = self._previous_dir(run_id)
        source = active if self._valid_generation(active, run_id) else None
        if source is None and self._valid_generation(previous, run_id):
            source = previous
        if source is None:
            if active.exists() or previous.exists():
                raise PersistenceError("no valid filesystem durable generation is recoverable")
            return False
        destination_dir.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file() and path.name != "durable_sync.json":
                rel = path.relative_to(source)
                dst = destination_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
        return True


class KagglePrivateDatasetStore(DurableStore):
    """Version a pre-created private Kaggle dataset with one run's minimal recovery bundle."""

    def __init__(self, dataset_slug: str, *, work_root: str | Path | None = None):
        if "/" not in dataset_slug:
            raise ValueError("Kaggle dataset slug must be owner/dataset")
        self.dataset_slug = dataset_slug
        self.work_root = Path(work_root) if work_root else None

    def _run(self, args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["KAGGLE_CONFIG_DIR"] = env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
        return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=1800)

    def sync(self, source_dir: Path, *, run_id: str, segment_id: str) -> SyncResult:
        root_ctx = tempfile.TemporaryDirectory(dir=self.work_root) if self.work_root else tempfile.TemporaryDirectory()
        with root_ctx as td:
            staging = Path(td)
            export_recovery_bundle(source_dir, staging)
            metadata = {
                "title": self.dataset_slug.split("/", 1)[1],
                "id": self.dataset_slug,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            }
            atomic_write_json(staging / "dataset-metadata.json", metadata)
            message = f"CropCop recovery {run_id} {segment_id}"
            self._run(["kaggle", "datasets", "version", "-p", str(staging), "-m", message, "-q", "-r", "zip"])
            files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
            return SyncResult("kaggle_private_dataset", self.dataset_slug, files)

    def restore(self, destination_dir: Path, *, run_id: str) -> bool:
        root_ctx = tempfile.TemporaryDirectory(dir=self.work_root) if self.work_root else tempfile.TemporaryDirectory()
        with root_ctx as td:
            staging = Path(td)
            try:
                self._run(["kaggle", "datasets", "download", "-d", self.dataset_slug, "-p", str(staging), "--unzip", "-q"])
            except subprocess.CalledProcessError as exc:
                raise PersistenceError(f"Kaggle recovery download failed: {exc.stderr[-1000:]}") from exc
            index = staging / "checkpoint_index.json"
            if not index.exists():
                return False
            destination_dir.mkdir(parents=True, exist_ok=True)
            for path in staging.rglob("*"):
                if path.is_file() and path.name != "dataset-metadata.json":
                    rel = path.relative_to(staging)
                    dst = destination_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)
            return True


def validate_durable_locator_template(kind: str, template: str, run_ids: list[str]) -> dict[str, str]:
    if kind not in {"filesystem", "kaggle-dataset"}:
        raise ValueError(f"unsupported durable store kind: {kind}")
    if not template:
        raise ValueError("durable locator template is empty")
    if kind == "kaggle-dataset" and "{run_id}" not in template and "{run_id_lower}" not in template:
        raise ValueError("Kaggle scientific durable locator must contain a run-ID placeholder")
    resolved = {rid: template.format(run_id=rid, run_id_lower=rid.lower()) for rid in run_ids}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("durable locator collision across run IDs")
    for rid, locator in resolved.items():
        if kind == "kaggle-dataset" and "/" not in locator:
            raise ValueError(f"invalid Kaggle durable locator for {rid}: {locator}")
    return resolved


def validate_durable_access_plan(
    kind: str,
    resolved: dict[str, str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fail-closed production durability preflight without mutating scientific generations.

    Filesystem stores receive an actual write/read/delete probe in every distinct root.
    Kaggle private datasets receive authenticated read probes and an exact owner binding;
    the first scientific/calibration sync remains the authoritative write exercise because
    creating a throwaway dataset version would itself mutate the durable history.
    """
    env = dict(os.environ if env is None else env)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if kind == "filesystem":
        roots = sorted({str(Path(locator).expanduser().resolve().parent) for locator in resolved.values()})
        for root_text in roots:
            root = Path(root_text)
            probe = root / f".cropcop-durable-probe-{os.getpid()}"
            try:
                root.mkdir(parents=True, exist_ok=True)
                payload = b"cropcop-durable-preflight\n"
                probe.write_bytes(payload)
                if probe.read_bytes() != payload:
                    raise PersistenceError("filesystem durability read-after-write probe mismatch")
                checks[root_text] = {"write": True, "read": True, "probe_removed": True}
            except Exception as exc:
                errors.append(f"filesystem durable access failed for {root}: {type(exc).__name__}: {exc}")
            finally:
                try:
                    probe.unlink(missing_ok=True)
                except Exception as exc:
                    errors.append(f"filesystem durable probe cleanup failed for {root}: {type(exc).__name__}: {exc}")

    elif kind == "kaggle-dataset":
        username = str(env.get("KAGGLE_USERNAME", "")).strip()
        key = str(env.get("KAGGLE_KEY", "")).strip()
        if not username or not key:
            errors.append("Kaggle durable access requires KAGGLE_USERNAME and KAGGLE_KEY")
        else:
            from .g1_publication import preflight_private_target

            cli_env = dict(env)
            cli_env["KAGGLE_CONFIG_DIR"] = cli_env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
            for run_id, locator in resolved.items():
                owner, _, dataset = locator.partition("/")
                row = {
                    "locator": locator,
                    "owner_matches_authenticated_user": owner.casefold() == username.casefold(),
                    "authenticated_read": False,
                    "authoritative_is_private": False,
                    "private_metadata_verified": False,
                    "write_authorization_basis": "private dataset owner identity",
                    "write_generation_mutated_by_preflight": False,
                }
                if not row["owner_matches_authenticated_user"]:
                    errors.append(
                        f"Kaggle durable locator owner mismatch for {run_id}: "
                        f"locator owner={owner!r}, authenticated user={username!r}"
                    )
                    checks[run_id] = row
                    continue
                if not dataset:
                    errors.append(f"invalid Kaggle durable locator for {run_id}: {locator}")
                    checks[run_id] = row
                    continue
                try:
                    private_state = preflight_private_target(locator, env=env)
                    row["authoritative_is_private"] = private_state.get("authoritative_is_private") is True
                    row["private_metadata_verified"] = True
                    row["current_version_number"] = private_state.get("current_version_number")
                except Exception as exc:
                    errors.append(
                        f"Kaggle durable private-target preflight failed for {run_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                try:
                    cp = subprocess.run(
                        ["kaggle", "datasets", "files", "-d", locator],
                        env=cli_env,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if cp.returncode != 0:
                        detail = (cp.stderr or cp.stdout or "").strip()[-800:]
                        errors.append(f"Kaggle durable read probe failed for {run_id}: {detail}")
                    else:
                        row["authenticated_read"] = True
                except Exception as exc:
                    errors.append(f"Kaggle durable read probe failed for {run_id}: {type(exc).__name__}: {exc}")
                checks[run_id] = row
    else:
        errors.append(f"unsupported durable-store kind: {kind}")

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "kind": kind,
        "resolved_locator_count": len(resolved),
        "checks": checks,
        "errors": errors,
        "note": (
            "Kaggle write authorization is bound to authenticated dataset ownership; "
            "the first real calibration/scientific sync is the non-fabricated write exercise."
            if kind == "kaggle-dataset" else
            "Filesystem access was exercised with an actual write/read/delete probe."
        ),
    }


def build_store(kind: str, locator: str) -> DurableStore:
    if kind == "filesystem":
        return FilesystemStore(locator)
    if kind == "kaggle-dataset":
        return KagglePrivateDatasetStore(locator)
    raise ValueError(f"unsupported durable store kind: {kind}")
