from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SENTINEL_NAME = "01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json"
AMENDMENT_NAME = "04A_EAAI_DUAL_GPU_EXECUTION_AMENDMENT_v1.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob(repo_root: Path, path: str) -> str:
    cp = subprocess.run(
        ["git", "hash-object", "--", path],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return cp.stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_science_diff(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    sentinel = _load(root / SENTINEL_NAME)
    errors: list[str] = []
    observed_blobs: dict[str, str] = {}
    expected_blobs = dict(sentinel.get("protected_git_blobs", {}))
    for path, expected in expected_blobs.items():
        full = root / path
        if not full.is_file():
            errors.append(f"protected scientific surface missing: {path}")
            continue
        observed = _git_blob(root, path)
        observed_blobs[path] = observed
        if observed != expected:
            errors.append(f"protected scientific surface drift: {path}: expected {expected}, got {observed}")

    authority = _load(root / "journal_extension/locks/scientific_authority.json")
    expected_authority = sentinel["authority"]
    if authority.get("authority_id") != expected_authority["stage03r_id"]:
        errors.append("Stage-03R authority ID drift")
    if authority.get("authority_sha256") != expected_authority["stage03r_sha256"]:
        errors.append("Stage-03R authority SHA-256 drift")
    if authority.get("stage04_architecture_id") != expected_authority["stage04_id"]:
        errors.append("Stage-04 base architecture ID drift")
    if authority.get("stage04_sha256") != expected_authority["stage04_sha256"]:
        errors.append("Stage-04 base architecture SHA-256 drift")

    amendment = root / AMENDMENT_NAME
    amendment_sha = _sha256_file(amendment) if amendment.is_file() else None
    if not amendment_sha:
        errors.append("Stage-04A execution amendment missing")
    else:
        if authority.get("stage04_execution_amendment_id") != "EAAI-JE-MGPU-A1":
            errors.append("Stage-04A execution amendment ID missing/mismatch")
        if authority.get("stage04_execution_amendment_sha256") != amendment_sha:
            errors.append("Stage-04A execution amendment SHA-256 mismatch")

    ctc = _load(root / "journal_extension/configs/common/ctc_v2.json")
    locked = sentinel["locked_training_semantics"]
    training = ctc.get("training", {})
    schedule = ctc.get("schedule", {})
    selection = ctc.get("selection", {})
    checks = {
        "micro_batch_size": training.get("micro_batch_size"),
        "gradient_accumulation": training.get("gradient_accumulation"),
        "effective_batch_size": training.get("effective_batch_size"),
        "validation_batch_size": training.get("validation_batch_size"),
        "epochs": schedule.get("epochs"),
        "mixed_precision": training.get("mixed_precision"),
        "gradient_clip_norm": training.get("gradient_clip_norm"),
        "ema": training.get("ema"),
        "drop_last": training.get("drop_last"),
        "sampling": training.get("sampling"),
        "early_stopping": selection.get("early_stopping"),
        "validation_selection_order": selection.get("order"),
    }
    for key, observed in checks.items():
        expected = locked.get(key)
        if observed != expected:
            errors.append(f"locked training semantic drift: {key}: expected {expected!r}, got {observed!r}")

    registry = _load(root / "journal_extension/locks/experiment_registry.json")
    by_id = {row["experiment_id"]: row for row in registry.get("experiments", [])}
    identities = sentinel["scientific_identities"]
    for seed_name, seed in identities["seeds"].items():
        pair = identities["pairs"][seed_name]
        for eid_key in ("direct", "teacher"):
            eid = pair[eid_key]
            row = by_id.get(eid)
            if not row:
                errors.append(f"frozen experiment missing: {eid}")
                continue
            if row.get("seed") != seed:
                errors.append(f"seed drift: {eid}")
            if row.get("pair_id") != pair["pair_id"]:
                errors.append(f"pair ID drift: {eid}")

    for condition, folder in (("direct", "r04_direct"), ("teacher", "r05_teacher")):
        for idx in (1, 2, 3):
            cfg = _load(root / f"journal_extension/configs/{folder}/s{idx}.json")
            if cfg.get("model_name") != identities["student_model"]:
                errors.append(f"student model identity drift: {folder}/s{idx}.json")
            expected_obj = identities["objectives"]["R04_direct" if condition == "direct" else "R05_teacher"]
            if cfg.get("objective") != expected_obj:
                errors.append(f"objective drift: {folder}/s{idx}.json")
            if condition == "teacher" and cfg.get("teacher_checkpoint_sha256") != identities["historical_teacher_sha256"]:
                errors.append(f"historical teacher identity drift: {folder}/s{idx}.json")

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "sentinel": SENTINEL_NAME,
        "stage03r_authority_unchanged": authority.get("authority_sha256") == expected_authority["stage03r_sha256"],
        "stage04_base_unchanged": authority.get("stage04_sha256") == expected_authority["stage04_sha256"],
        "stage04_execution_amendment_sha256": amendment_sha,
        "protected_surfaces_unchanged": not any("protected scientific surface" in e for e in errors),
        "observed_git_blobs": observed_blobs,
        "errors": errors,
    }
