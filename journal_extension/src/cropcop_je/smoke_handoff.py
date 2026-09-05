from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json, fsync_directory
from .checkpointing import read_index
from .hashing import sha256_file, sha256_json

SMOKE_MANIFEST_SCHEMA = "1.0"
SMOKE_A_MANIFEST = "SMOKE_A_MANIFEST.json"
SMOKE_A_EVIDENCE = "SMOKE_A_EVIDENCE.json"
SMOKE_B_MANIFEST = "SMOKE_B_MANIFEST.json"
SMOKE_B_EVIDENCE = "SMOKE_B_EVIDENCE.json"


class SmokeHandoffError(RuntimeError):
    pass


QUALIFYING_KAGGLE_RUN_TYPES = {"Batch"}
TERMINAL_SMOKE_B_SEQUENCE = (
    "READ_A", "VERIFY_A", "RESTORE_A", "RECOVER_A",
    "LOAD_A", "RESUME", "CHECKPOINT_B",
)


def observed_kaggle_run_type(evidence: dict[str, Any] | None = None) -> str:
    evidence = evidence or {}
    explicit = str(evidence.get("kaggle_run_type", "")).strip()
    if explicit:
        return explicit
    nested = evidence.get("environment", {}).get("kaggle", {})
    nested_value = str(nested.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip()
    if nested_value:
        return nested_value
    return str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip()


def require_qualifying_kaggle_batch(*, context: str, run_type: str | None = None) -> str:
    observed = (run_type if run_type is not None else os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip()
    if observed not in QUALIFYING_KAGGLE_RUN_TYPES:
        label = observed or "<missing>"
        raise SmokeHandoffError(
            f"{context} requires clean Kaggle Saved-Version/Batch execution; "
            f"observed KAGGLE_KERNEL_RUN_TYPE={label}. Interactive execution is DIAGNOSTIC_NOT_QUALIFYING."
        )
    return observed


def validate_terminal_smoke_b_evidence(
    evidence: dict[str, Any],
    *,
    expected_source_sha: str,
    expected_dependency_lock_sha256: str,
    require_batch: bool = True,
) -> list[str]:
    errors: list[str] = []
    if evidence.get("schema_version") != "2.0":
        errors.append("unsupported Smoke-B evidence schema")
    if evidence.get("status") != "PASS":
        errors.append("Smoke-B status is not PASS")
    if evidence.get("mode") != "RESTORE":
        errors.append("Smoke-B mode is not RESTORE")
    if evidence.get("scientific") is not False:
        errors.append("Smoke-B is not explicitly non-scientific")
    if evidence.get("synthetic_unprotected_data_only") is not True:
        errors.append("Smoke-B is not synthetic/unprotected-only")
    if evidence.get("source_git_sha") != expected_source_sha:
        errors.append("Smoke-B source Git SHA mismatch")
    if evidence.get("dependency_lock_sha256") != expected_dependency_lock_sha256:
        errors.append("Smoke-B dependency-lock SHA mismatch")

    run_type = observed_kaggle_run_type(evidence)
    if require_batch and run_type not in QUALIFYING_KAGGLE_RUN_TYPES:
        errors.append(
            "Smoke-B terminal qualification requires clean Kaggle Saved-Version/Batch execution "
            f"(observed {run_type or '<missing>'})"
        )

    for field in ("restore_success", "recover_success", "resume_success"):
        if evidence.get(field) is not True:
            errors.append(f"Smoke-B {field} is not true")

    if tuple(evidence.get("sequence") or ()) != TERMINAL_SMOKE_B_SEQUENCE:
        errors.append("Smoke-B restore/recover/resume sequence mismatch")

    expected_checkpoint = str(evidence.get("smoke_a_expected_checkpoint_sha256", ""))
    restored_checkpoint = str(evidence.get("smoke_a_observed_restored_checkpoint_sha256", ""))
    if len(expected_checkpoint) != 64 or len(restored_checkpoint) != 64:
        errors.append("Smoke-B Smoke-A checkpoint SHA reference missing/invalid")
    elif expected_checkpoint != restored_checkpoint:
        errors.append("Smoke-B restored checkpoint SHA differs from expected Smoke-A checkpoint SHA")

    try:
        restored_step = int(evidence.get("restored_optimizer_step", -1))
        resumed_step = int(evidence.get("resumed_optimizer_step", -1))
        if restored_step < 0 or resumed_step <= restored_step:
            errors.append("Smoke-B optimizer step did not advance beyond restored step")
    except (TypeError, ValueError):
        errors.append("Smoke-B optimizer step fields are invalid")

    if evidence.get("attached_smoke_a_input_unchanged") is not True:
        errors.append("Smoke-B attached Smoke-A input was not proven unchanged")

    for field in ("g1_executed", "g2_executed", "r04_r05_executed"):
        if evidence.get(field) is not False:
            errors.append(f"Smoke-B forbidden execution flag is not false: {field}")
    if evidence.get("restricted_cropcop_data_accessed") is not False:
        errors.append("Smoke-B accessed or did not explicitly exclude restricted CropCop data")

    if evidence.get("git_publication_status") != "PASS":
        errors.append("Smoke-B public-safe evidence publication is not PASS")
    if not str(evidence.get("public_safe_evidence_branch", "")).startswith("run-evidence/SMOKE-B-"):
        errors.append("Smoke-B public-safe evidence branch reference missing/invalid")

    for field in ("smoke_a_manifest_sha256", "smoke_a_evidence_sha256"):
        if len(str(evidence.get(field, ""))) != 64:
            errors.append(f"Smoke-B {field} missing/invalid")

    return errors


def _hash_without(payload: dict[str, Any], field: str) -> str:
    clean = dict(payload)
    clean.pop(field, None)
    return sha256_json(clean)


def manifest_self_hash(payload: dict[str, Any]) -> str:
    return _hash_without(payload, "manifest_sha256")


def safe_relative_path(path: str) -> Path:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise SmokeHandoffError(f"unsafe relative smoke-export path: {path!r}")
    return rel


def file_record(path: str | Path, root: str | Path) -> dict[str, Any]:
    p = Path(path)
    r = Path(root)
    rel = p.relative_to(r).as_posix()
    return {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}


def snapshot_files(root: str | Path) -> dict[str, dict[str, Any]]:
    r = Path(root)
    return {
        p.relative_to(r).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(r.rglob("*"))
        if p.is_file()
    }


def recovery_file_records(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    index = read_index(root)
    refs = [index.get("latest"), index.get("previous"), index.get("selected")]
    paths = {Path("checkpoint_index.json")}
    for ref in refs:
        if ref:
            paths.add(safe_relative_path(str(ref["relative_path"])))
    records = []
    for rel in sorted(paths, key=lambda p: p.as_posix()):
        path = root / rel
        if not path.is_file():
            raise SmokeHandoffError(f"recovery file missing before export finalization: {rel}")
        records.append(file_record(path, root))
    return records


def verify_records(root: str | Path, records: list[dict[str, Any]]) -> None:
    root = Path(root).resolve()
    if not records:
        raise SmokeHandoffError("smoke export contains no recovery-file records")
    seen: set[str] = set()
    for row in records:
        rel = safe_relative_path(str(row.get("path", "")))
        key = rel.as_posix()
        if key in seen:
            raise SmokeHandoffError(f"duplicate smoke-export file record: {key}")
        seen.add(key)
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            raise SmokeHandoffError(f"smoke-export path escapes root: {key}")
        if not path.is_file():
            raise SmokeHandoffError(f"smoke-export file missing: {key}")
        if path.stat().st_size != int(row.get("bytes", -1)):
            raise SmokeHandoffError(f"smoke-export byte count mismatch: {key}")
        if sha256_file(path) != row.get("sha256"):
            raise SmokeHandoffError(f"smoke-export SHA-256 mismatch: {key}")


def build_smoke_a_manifest(
    *,
    qualification_id: str,
    source_git_sha: str,
    source_tree_sha: str,
    dependency_lock_sha256: str,
    environment_identity_sha256: str,
    checkpoint_relative_path: str,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    checkpoint_identity_sha256: str,
    optimizer_step: int,
    recovery_files: list[dict[str, Any]],
    smoke_a_evidence_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": SMOKE_MANIFEST_SCHEMA,
        "qualification_id": qualification_id,
        "mode": "WRITE",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "dependency_lock_sha256": dependency_lock_sha256,
        "environment_identity_sha256": environment_identity_sha256,
        "optimizer_step": int(optimizer_step),
        "checkpoint_relative_path": safe_relative_path(checkpoint_relative_path).as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": int(checkpoint_bytes),
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "recovery_files": recovery_files,
        "smoke_a_evidence_sha256": smoke_a_evidence_sha256,
        "created_at_utc": created_at_utc,
    }
    payload["manifest_sha256"] = manifest_self_hash(payload)
    return payload


def validate_smoke_a_manifest(
    manifest: dict[str, Any],
    *,
    expected_source_sha: str,
    expected_dependency_lock_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SMOKE_MANIFEST_SCHEMA:
        errors.append("unsupported Smoke-A manifest schema")
    if manifest.get("mode") != "WRITE":
        errors.append("Smoke-A manifest mode is not WRITE")
    if manifest.get("scientific") is not False:
        errors.append("Smoke-A manifest is not explicitly non-scientific")
    if manifest.get("synthetic_unprotected_data_only") is not True:
        errors.append("Smoke-A manifest is not synthetic/unprotected-only")
    qid = str(manifest.get("qualification_id", ""))
    if not qid.startswith("INFRA-SMOKE-") or len(qid) < 24:
        errors.append("Smoke-A qualification ID missing/invalid")
    if manifest.get("source_git_sha") != expected_source_sha:
        errors.append("Smoke-A source Git SHA mismatch")
    if manifest.get("dependency_lock_sha256") != expected_dependency_lock_sha256:
        errors.append("Smoke-A dependency-lock SHA mismatch")
    for field in (
        "source_tree_sha", "environment_identity_sha256", "checkpoint_sha256",
        "checkpoint_identity_sha256", "smoke_a_evidence_sha256", "manifest_sha256",
    ):
        if len(str(manifest.get(field, ""))) != 64 and field != "source_tree_sha":
            errors.append(f"Smoke-A {field} missing/invalid")
    if len(str(manifest.get("source_tree_sha", ""))) != 40:
        errors.append("Smoke-A source tree SHA missing/invalid")
    if int(manifest.get("checkpoint_bytes", 0) or 0) <= 0:
        errors.append("Smoke-A checkpoint byte count invalid")
    if int(manifest.get("optimizer_step", -1)) < 1:
        errors.append("Smoke-A optimizer step invalid")
    try:
        safe_relative_path(str(manifest.get("checkpoint_relative_path", "")))
    except Exception as exc:
        errors.append(str(exc))
    if manifest.get("manifest_sha256") != manifest_self_hash(manifest):
        errors.append("Smoke-A manifest self-hash mismatch")
    if not isinstance(manifest.get("recovery_files"), list) or not manifest.get("recovery_files"):
        errors.append("Smoke-A recovery file list missing")
    return errors


@dataclass(frozen=True)
class VerifiedSmokeA:
    export_root: Path
    manifest_path: Path
    evidence_path: Path
    checkpoint_path: Path
    manifest: dict[str, Any]
    evidence: dict[str, Any]


def locate_smoke_a_export(input_root: str | Path) -> Path:
    root = Path(input_root)
    if not root.is_dir():
        raise SmokeHandoffError(f"SMOKE_A_INPUT_ROOT does not exist or is not a directory: {root}")
    matches = sorted(root.rglob(SMOKE_A_MANIFEST))
    if len(matches) != 1:
        raise SmokeHandoffError(
            f"SMOKE_A_INPUT_ROOT must contain exactly one {SMOKE_A_MANIFEST}; found {len(matches)}"
        )
    return matches[0].parent


def verify_smoke_a_export(
    input_root: str | Path,
    *,
    expected_source_sha: str,
    expected_dependency_lock_sha256: str,
) -> VerifiedSmokeA:
    export = locate_smoke_a_export(input_root)
    manifest_path = export / SMOKE_A_MANIFEST
    evidence_path = export / SMOKE_A_EVIDENCE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_smoke_a_manifest(
        manifest,
        expected_source_sha=expected_source_sha,
        expected_dependency_lock_sha256=expected_dependency_lock_sha256,
    )
    if errors:
        raise SmokeHandoffError("invalid Smoke-A manifest: " + "; ".join(errors))
    if not evidence_path.is_file():
        raise SmokeHandoffError(f"{SMOKE_A_EVIDENCE} is missing")
    if sha256_file(evidence_path) != manifest.get("smoke_a_evidence_sha256"):
        raise SmokeHandoffError("Smoke-A evidence SHA differs from manifest")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("qualification_id", manifest["qualification_id"]),
        ("source_git_sha", expected_source_sha),
        ("dependency_lock_sha256", expected_dependency_lock_sha256),
    ):
        if evidence.get(field) != expected:
            raise SmokeHandoffError(f"Smoke-A evidence {field} differs from manifest/authorized value")
    if evidence.get("scientific") is not False or evidence.get("synthetic_unprotected_data_only") is not True:
        raise SmokeHandoffError("Smoke-A evidence does not preserve non-scientific/synthetic-only flags")
    verify_records(export, list(manifest["recovery_files"]))
    checkpoint = export / safe_relative_path(manifest["checkpoint_relative_path"])
    if checkpoint.stat().st_size != int(manifest["checkpoint_bytes"]):
        raise SmokeHandoffError("Smoke-A checkpoint byte count differs from manifest")
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise SmokeHandoffError("Smoke-A checkpoint SHA differs from manifest")
    index = read_index(export)
    latest = index.get("latest")
    if not latest or latest.get("sha256") != manifest["checkpoint_sha256"]:
        raise SmokeHandoffError("Smoke-A checkpoint index does not bind manifest checkpoint SHA")
    if latest.get("identity_sha256") != manifest["checkpoint_identity_sha256"]:
        raise SmokeHandoffError("Smoke-A checkpoint index identity digest differs from manifest")
    if int(latest.get("optimizer_step", -1)) != int(manifest["optimizer_step"]):
        raise SmokeHandoffError("Smoke-A checkpoint index optimizer step differs from manifest")
    return VerifiedSmokeA(export, manifest_path, evidence_path, checkpoint, manifest, evidence)


def restore_verified_recovery_bundle(verified: VerifiedSmokeA, destination: str | Path) -> Path:
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise SmokeHandoffError(f"Smoke-B restore destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for row in verified.manifest["recovery_files"]:
        rel = safe_relative_path(row["path"])
        src = verified.export_root / rel
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    fsync_directory(destination)
    verify_records(destination, list(verified.manifest["recovery_files"]))
    return destination


class RestoreSequence:
    ORDER = ("READ_A", "VERIFY_A", "RESTORE_A", "RECOVER_A", "LOAD_A", "RESUME", "CHECKPOINT_B")

    def __init__(self) -> None:
        self.events: list[str] = []

    def advance(self, event: str) -> None:
        expected = self.ORDER[len(self.events)] if len(self.events) < len(self.ORDER) else None
        if event != expected:
            raise SmokeHandoffError(f"invalid Smoke-B sequence: expected {expected}, got {event}")
        self.events.append(event)

    @property
    def recovery_verified(self) -> bool:
        return self.events[:5] == list(self.ORDER[:5])

    @property
    def resume_verified(self) -> bool:
        return self.events[:6] == list(self.ORDER[:6])

    def require_resume_before_checkpoint(self) -> None:
        if not self.resume_verified:
            raise SmokeHandoffError("Smoke-B checkpoint/output write forbidden before verified recovery + resume")
