from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import require_sha256, sha256_file, sha256_json
from .models import build_projection_without_state_drift, load_exact_teacher, load_pair_initialization
from .secondary import load_baseline_initialization
from .tracka_v12 import EXPERIMENT_SPECS, PAIR_IDS, validate_tracka_v12_config
from .tracka_v12_g1a import (
    R13_PRETRAINED_SHA256,
    TEACHER_SHA256,
    load_r13_initialization,
    validate_g1a_seal_object,
)


class TrackAV12RuntimeError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_and_validate_g1a_bundle(
    bundle_dir: str | Path,
    *,
    expected_source_sha: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    root = Path(bundle_dir)
    seal_path = root / "TRACKA_V12_G1A_SEAL.json"
    if not seal_path.is_file():
        return {}, ["Track-A v1.2 G1A seal missing"]
    seal = load_json(seal_path)
    errors = validate_g1a_seal_object(seal)
    if expected_source_sha and seal.get("source_git_sha") != expected_source_sha:
        errors.append("G1A source SHA differs from requested execution source")
    private = root / "private"
    evidence = root / "evidence"

    for label in ("S2", "S3"):
        row = seal.get("r12_reused_pairs", {}).get(label, {})
        path = private / str(row.get("student_init_basename", ""))
        expected_sha = str(row.get("student_init_sha256", ""))
        if not path.is_file() or sha256_file(path) != expected_sha:
            errors.append(f"G1A R12 {label} pair bytes mismatch")
        ev = evidence / f"R12_REUSED_PAIR_{label}.json"
        if not ev.is_file() or sha256_json(load_json(ev)) != row.get("reuse_evidence_sha256"):
            errors.append(f"G1A R12 {label} reuse evidence mismatch")

    teacher = seal.get("teacher", {})
    teacher_path = private / str(teacher.get("artifact_basename", ""))
    if not teacher_path.is_file() or sha256_file(teacher_path) != TEACHER_SHA256:
        errors.append("G1A teacher bytes mismatch")
    for basename, expected_sha in teacher.get("evidence_sha256", {}).items():
        path = evidence / basename
        if not path.is_file() or sha256_json(load_json(path)) != expected_sha:
            errors.append(f"G1A teacher evidence mismatch: {basename}")

    for key in ("effb0", "cnxtt"):
        for label in ("S2", "S3"):
            row = seal.get("baselines", {}).get(key, {}).get(label, {})
            pretrained = private / str(row.get("pretrained_basename", ""))
            init = private / str(row.get("init_basename", ""))
            if not pretrained.is_file() or sha256_file(pretrained) != row.get("pretrained_sha256"):
                errors.append(f"G1A {key} pretrained bytes mismatch")
            if not init.is_file() or sha256_file(init) != row.get("init_sha256"):
                errors.append(f"G1A {key} {label} init bytes mismatch")
            ev = evidence / f"{key.upper()}_INIT_{label}.json"
            if not ev.is_file() or sha256_json(load_json(ev)) != row.get("init_evidence_sha256"):
                errors.append(f"G1A {key} {label} init evidence mismatch")

    r13 = seal.get("r13", {})
    pretrained = private / str(r13.get("pretrained_basename", ""))
    if not pretrained.is_file() or sha256_file(pretrained) != R13_PRETRAINED_SHA256:
        errors.append("G1A R13 upstream bytes mismatch")
    parity = evidence / "R13_NORMALIZATION_PARITY.json"
    if not parity.is_file() or sha256_json(load_json(parity)) != r13.get("parity_evidence_sha256"):
        errors.append("G1A R13 parity evidence mismatch")
    for label in ("S1", "S2", "S3"):
        row = r13.get("states", {}).get(label, {})
        init = private / str(row.get("init_basename", ""))
        if not init.is_file() or sha256_file(init) != row.get("init_sha256"):
            errors.append(f"G1A R13 {label} init bytes mismatch")
        ev = evidence / f"R13_VIT_DLITTLE_INIT_{label}.json"
        if not ev.is_file() or sha256_json(load_json(ev)) != row.get("init_evidence_sha256"):
            errors.append(f"G1A R13 {label} init evidence mismatch")
    return seal, errors


def _seed_label(experiment_id: str) -> str:
    spec = EXPERIMENT_SPECS.get(experiment_id)
    if spec is None:
        raise TrackAV12RuntimeError(f"unauthorized Track-A v1.2 experiment: {experiment_id}")
    return str(spec["seed_label"])


def load_student_and_teacher(
    *,
    repo_root: str | Path,
    config: dict[str, Any],
    g1a_bundle: str | Path,
    expected_source_sha: str,
):
    errors = validate_tracka_v12_config(config)
    if errors:
        raise TrackAV12RuntimeError("Track-A v1.2 config invalid: " + "; ".join(errors))
    experiment_id = str(config["experiment_id"])
    spec = EXPERIMENT_SPECS[experiment_id]
    seed = int(spec["seed"])
    label = _seed_label(experiment_id)
    root = Path(g1a_bundle)
    seal, bundle_errors = load_and_validate_g1a_bundle(root, expected_source_sha=expected_source_sha)
    if bundle_errors:
        raise TrackAV12RuntimeError("Track-A v1.2 G1A bundle invalid: " + "; ".join(bundle_errors))
    private = root / "private"
    evidence = root / "evidence"
    family = str(spec["model_family"])
    student = None
    teacher = None
    projection = None
    teacher_identity = None
    teacher_factory_bundle_sha = None
    teacher_sha = None

    if family == "mnv4":
        row = seal["r12_reused_pairs"][label]
        if experiment_id not in set(row["authorized_consumers"]):
            raise TrackAV12RuntimeError("R12 experiment is not authorized for selected reused pair")
        student, payload = load_pair_initialization(
            private / row["student_init_basename"],
            expected_sha256=row["student_init_sha256"],
            pair_id=PAIR_IDS[label],
            seed=seed,
        )
        if payload.get("pretrained_sha256") != config.get("required_pretrained_evidence") and False:
            raise TrackAV12RuntimeError("unreachable compatibility guard")
        init_sha = row["student_init_sha256"]
        pretrained_sha = payload["pretrained_sha256"]
        teacher_row = seal["teacher"]
        factory_manifest = load_json(evidence / "TEACHER_FACTORY_BUNDLE.json")
        factory_spec = str(factory_manifest.get("entrypoint", ""))
        if not factory_spec or ":" not in factory_spec:
            raise TrackAV12RuntimeError("sealed teacher factory entrypoint missing")
        teacher, teacher_identity = load_exact_teacher(
            private / teacher_row["artifact_basename"],
            factory_spec=factory_spec,
            factory_bundle_manifest=evidence / "TEACHER_FACTORY_BUNDLE.json",
            repo_root=repo_root,
            factory_source_root=Path(repo_root),
        )
        teacher_sha = TEACHER_SHA256
        teacher_factory_bundle_sha = teacher_identity["bundle_sha256"]
        if float(config["objective"].get("feature", 0.0)) > 0:
            projection = build_projection_without_state_drift(student, teacher, seed=seed)
    elif family in {"effb0", "cnxtt"}:
        row = seal["baselines"][family][label]
        if row.get("authorized_consumers") != [experiment_id]:
            raise TrackAV12RuntimeError("baseline initialization consumer mismatch")
        student, payload = load_baseline_initialization(
            private / row["init_basename"],
            expected_sha256=row["init_sha256"],
            model_key=family,
            seed=seed,
            experiment_id=experiment_id,
        )
        init_sha = row["init_sha256"]
        pretrained_sha = row["pretrained_sha256"]
        if payload.get("pretrained_sha256") != pretrained_sha:
            raise TrackAV12RuntimeError("baseline initialization/pretrained identity mismatch")
    elif family == "vit_dlittle_diff":
        row = seal["r13"]["states"][label]
        if row.get("authorized_consumers") != [experiment_id]:
            raise TrackAV12RuntimeError("R13 initialization consumer mismatch")
        student, payload = load_r13_initialization(
            private / row["init_basename"],
            expected_sha256=row["init_sha256"],
            experiment_id=experiment_id,
            seed=seed,
        )
        init_sha = row["init_sha256"]
        pretrained_sha = payload["pretrained_sha256"]
    else:
        raise TrackAV12RuntimeError(f"unsupported Track-A v1.2 model family: {family}")

    return {
        "student": student,
        "teacher": teacher,
        "projection": projection,
        "student_init_sha256": init_sha,
        "pretrained_sha256": pretrained_sha,
        "teacher_sha256": teacher_sha,
        "teacher_factory_sha256": sha256_json(teacher_identity) if teacher_identity else None,
        "teacher_factory_bundle_sha256": teacher_factory_bundle_sha,
        "g1a_seal_sha256": seal["g1a_seal_sha256"],
        "g1a_seal": seal,
    }
