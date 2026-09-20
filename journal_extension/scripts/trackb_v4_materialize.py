from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
import hashlib
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.g1 import validate_teacher_factory_bundle
from cropcop_je.models import load_exact_teacher
from cropcop_je.trackb_r07 import (
    CLASS_MAP_SHA256,
    DATASET_MANIFEST_SHA256,
    DINO_AUDIT_SHA256,
    DINO_FACTORY_MANIFEST_SHA256,
    R07_CHECKPOINTS,
    R07_RUN_RECORDS,
    TrackBError,
    load_json,
)
from cropcop_je.trackb_r07_ops import (
    TrackBOpsError,
    acquire_gvlid_v5,
    acquire_irish_potato,
    configure_runtime_secrets,
    probe_external_sources,
    publish_private_kaggle_dataset,
    run_checked,
    verify_authenticated_kaggle_owner,
)

SOURCE_SLUG_BASENAMES = {
    "final_v1": "cropcop-finalized-v8-11-2026-1",
    "r07_s1": "sec-je-r07-cnxtt-context-s1-8904b100d223-a01",
    "r07_s2": "cropcop-r07-cnxtt-context-s2-abce1197-56023042",
    "r07_s3": "cropcop-r07-cnxtt-context-s3-f13ca687-56023042",
    "dino_bundle": "cropcop-secondary-g1-8904b100",
}

INFRA_DATASET_PREFIX = "cropcop-trackb-infrastructure-v5"
EXTERNAL_DATASET_PREFIX = "cropcop-trackb-external-v5"


def stage(name: str) -> None:
    print(f"\n{'=' * 18} {name} {'=' * 18}", flush=True)


def _resolve_attached_root(input_root: Path, basename: str) -> Path:
    candidates: list[Path] = []
    direct = input_root / basename
    if direct.is_dir():
        candidates.append(direct.resolve())
    for path in input_root.rglob(basename):
        if path.is_dir():
            resolved = path.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    if len(candidates) != 1:
        raise TrackBOpsError(
            f"expected exactly one attached Kaggle dataset mount named {basename!r}; "
            f"found {[str(p) for p in candidates]}"
        )
    return candidates[0]


def _find_v1(root: Path) -> tuple[Path, Path, Path]:
    manifest_matches = [
        path.resolve()
        for path in root.rglob("final_manifest.csv")
        if path.is_file() and sha256_file(path) == DATASET_MANIFEST_SHA256
    ]
    class_matches = [
        path.resolve()
        for path in root.rglob("class_to_idx.json")
        if path.is_file() and sha256_file(path) == CLASS_MAP_SHA256
    ]

    resolved: list[tuple[Path, Path, Path]] = []
    diagnostics: list[dict[str, object]] = []
    for manifest in sorted(manifest_matches):
        local_class_maps = sorted(
            path for path in class_matches if path.parent == manifest.parent
        )
        roots: list[Path] = []
        for parent in (
            manifest.parent.parent,
            manifest.parent,
            manifest.parent.parent.parent,
        ):
            image_root = parent / "dataset"
            if (image_root / "train").is_dir() and (image_root / "val").is_dir():
                roots.append(image_root.resolve())
        roots = list(dict.fromkeys(roots))
        diagnostics.append({
            "manifest": str(manifest),
            "same_directory_class_maps": [str(path) for path in local_class_maps],
            "image_roots": [str(path) for path in roots],
        })
        if len(local_class_maps) == 1 and len(roots) == 1:
            resolved.append((manifest, local_class_maps[0], roots[0]))

    unique: list[tuple[Path, Path, Path]] = []
    seen: set[tuple[str, str, str]] = set()
    for manifest, class_map, image_root in resolved:
        key = (str(manifest), str(class_map), str(image_root))
        if key not in seen:
            seen.add(key)
            unique.append((manifest, class_map, image_root))

    if len(unique) != 1:
        raise TrackBOpsError(
            "could not resolve exactly one image-backed frozen Final-V1 authority pair; "
            + json.dumps({
                "manifest_hash_matches": [str(path) for path in manifest_matches],
                "class_map_hash_matches": [str(path) for path in class_matches],
                "structural_candidates": diagnostics,
            }, sort_keys=True)
        )
    return unique[0]


def _prepare_final_v1_core_view(source_root: Path, view_root: Path) -> Path:
    manifest, class_map, image_root = _find_v1(source_root)
    if view_root.exists():
        shutil.rmtree(view_root)
    audit = view_root / "audit"
    audit.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, audit / "final_manifest.csv")
    shutil.copy2(class_map, audit / "class_to_idx.json")

    dataset_link = view_root / "dataset"
    dataset_link.symlink_to(image_root, target_is_directory=True)

    if sha256_file(audit / "final_manifest.csv") != DATASET_MANIFEST_SHA256:
        raise TrackBOpsError("canonical Final-V1 view manifest hash drift")
    if sha256_file(audit / "class_to_idx.json") != CLASS_MAP_SHA256:
        raise TrackBOpsError("canonical Final-V1 view class-map hash drift")
    if not (dataset_link / "train").is_dir() or not (dataset_link / "val").is_dir():
        raise TrackBOpsError("canonical Final-V1 view does not resolve train/val image roots")
    return view_root



def _find_exact_file_by_sha(
    root: Path,
    *,
    expected_sha256: str,
    suffixes: set[str] | None = None,
    max_bytes: int | None = None,
) -> Path:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if max_bytes is not None and size > max_bytes:
            continue
        if sha256_file(path) == expected_sha256:
            matches.append(path.resolve())
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise TrackBOpsError(
            f"expected exactly one file with SHA-256 {expected_sha256} under {root}; "
            f"found {[str(path) for path in matches]}"
        )
    return matches[0]


def _selected_checkpoint_from_index(root: Path, expected_sha256: str) -> tuple[Path, Path]:
    candidates: list[tuple[Path, Path]] = []
    for index_path in sorted(root.rglob("checkpoint_index.json")):
        if not index_path.is_file():
            continue
        try:
            payload = load_json(index_path)
        except Exception:
            continue
        selected = payload.get("selected")
        if not isinstance(selected, dict):
            continue
        if str(selected.get("sha256", "")) != expected_sha256:
            continue
        rel = str(selected.get("relative_path", "")).strip()
        if not rel:
            continue
        selected_path = (index_path.parent / rel).resolve()
        if not selected_path.is_file():
            continue
        if sha256_file(selected_path) != expected_sha256:
            continue
        candidates.append((index_path.parent.resolve(), selected_path))
    unique: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for root_path, selected_path in candidates:
        key = (str(root_path), str(selected_path))
        if key not in seen:
            seen.add(key)
            unique.append((root_path, selected_path))
    if len(unique) != 1:
        raise TrackBOpsError(
            f"expected exactly one checkpoint_index.json binding selected checkpoint "
            f"{expected_sha256} under {root}; found "
            f"{[(str(a), str(b)) for a, b in unique]}"
        )
    return unique[0]


def _symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _verify_run_record_contract(path: Path, seed: str) -> dict:
    observed_sha = sha256_file(path)
    expected = R07_RUN_RECORDS[seed]
    if observed_sha != expected["sha256"]:
        raise TrackBOpsError(
            f"R07 {seed} run-record SHA mismatch: expected {expected['sha256']}, got {observed_sha}"
        )
    record = load_json(path)
    if str(record.get("run_id", "")) != expected["run_id"]:
        raise TrackBOpsError(f"R07 {seed} run-record ID mismatch")
    selected_sha = (
        ((record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
        or (record.get("result_summary") or {}).get("selected_checkpoint_sha256")
        or record.get("selected_checkpoint_sha256")
    )
    if str(selected_sha) != R07_CHECKPOINTS[seed]:
        raise TrackBOpsError(f"R07 {seed} run record does not bind the frozen checkpoint")
    selected_metrics = (record.get("result_summary") or {}).get("selected_metrics") or {}
    required_metrics = {
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "validation_nll",
    }
    if not required_metrics.issubset(selected_metrics):
        raise TrackBOpsError(
            f"R07 {seed} run record lacks frozen replay metrics: "
            f"{sorted(required_metrics.difference(selected_metrics))}"
        )
    return record


def _prepare_r07_source_views(
    repo_root: Path,
    mounts: dict[str, Path],
    source_views: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    authority_root = repo_root / "journal_extension/track_b_r07/replay_authority"
    s1_record = authority_root / "R07_S1_ORIGINAL_RUN_RECORD.json"
    k3_report = authority_root / "TRACKA_V12_K3_PUBLIC_REPORT.json"
    if not s1_record.is_file() or not k3_report.is_file():
        raise TrackBOpsError("Track-B replay-authority files are missing from the frozen repository snapshot")

    _verify_run_record_contract(s1_record, "S1")
    k3 = load_json(k3_report)
    if (
        k3.get("status") != "PASS"
        or k3.get("science_complete") is not True
        or k3.get("science_source_sha") != "56023042e57758591df9babb3438f191dbe10312"
        or k3.get("science_authorization_sha256") != "58d65a9c9f0c06222c00541feca9b2aaa005ebda850d2261d4df3e0e1fe7fbb7"
        or k3.get("scheduler_freeze_sha256") != "7c7eba73096c13eb7fa754143b9fe84548c16ac05ab80f6aa3e1aeeb128c3787"
    ):
        raise TrackBOpsError("frozen K3 terminal account report identity/status mismatch")

    views: dict[str, Path] = {}
    evidence: dict[str, object] = {}

    s1_checkpoint = _find_exact_file_by_sha(
        mounts["r07_s1"],
        expected_sha256=R07_CHECKPOINTS["S1"],
        suffixes={".pt", ".pth", ".ckpt", ".bin", ".safetensors", ""},
    )
    s1_view = source_views / "r07_s1"
    s1_view.mkdir(parents=True, exist_ok=False)
    _symlink_file(s1_checkpoint, s1_view / "selected_checkpoint.pt")
    shutil.copy2(s1_record, s1_view / "run_record.json")
    views["r07_s1"] = s1_view
    evidence["S1"] = {
        "record_kind": "ORIGINAL_PUBLIC_RUN_RECORD",
        "run_record_sha256": sha256_file(s1_view / "run_record.json"),
        "run_id": R07_RUN_RECORDS["S1"]["run_id"],
        "checkpoint_sha256": sha256_file(s1_checkpoint),
        "checkpoint_source_path": str(s1_checkpoint),
    }

    continuation = {
        "S2": {
            "mount": mounts["r07_s2"],
            "experiment_id": "R07-CNXTT-CONTEXT-S2",
            "durable_locator": "sabahatabbas/cropcop-r07-cnxtt-context-s2-abce1197-56023042",
        },
        "S3": {
            "mount": mounts["r07_s3"],
            "experiment_id": "R07-CNXTT-CONTEXT-S3",
            "durable_locator": "sabahatabbas/cropcop-r07-cnxtt-context-s3-f13ca687-56023042",
        },
    }

    recovery_script = repo_root / "journal_extension/scripts/recover_tracka_v12_terminal_record.py"
    for seed, spec in continuation.items():
        restored_from_durable = False
        try:
            checkpoint_root, selected_checkpoint = _selected_checkpoint_from_index(
                spec["mount"], R07_CHECKPOINTS[seed]
            )
        except TrackBOpsError:
            checkpoint_root = source_views / "_restored_checkpoints" / seed.lower()
            checkpoint_root.mkdir(parents=True, exist_ok=False)
            selected_checkpoint = None
            restored_from_durable = True

        recovery_out = source_views / "_recovery" / seed.lower()
        run_checked(
            [
                sys.executable,
                str(recovery_script),
                "--repo-root", str(repo_root),
                "--experiment-id", spec["experiment_id"],
                "--run-id", R07_RUN_RECORDS[seed]["run_id"],
                "--account-report", str(k3_report),
                "--checkpoint-root", str(checkpoint_root),
                "--durable-store-locator", spec["durable_locator"],
                "--output-dir", str(recovery_out),
            ],
            cwd=repo_root,
            timeout=3600,
        )
        if restored_from_durable:
            checkpoint_root, selected_checkpoint = _selected_checkpoint_from_index(
                checkpoint_root, R07_CHECKPOINTS[seed]
            )
        assert selected_checkpoint is not None
        recovered = recovery_out / "RECOVERED_TERMINAL_RUN_RECORD.json"
        certificate = recovery_out / "TERMINAL_RECOVERY_CERTIFICATE.json"
        if not recovered.is_file() or not certificate.is_file():
            raise TrackBOpsError(f"R07 {seed} terminal recovery did not produce required artifacts")
        _verify_run_record_contract(recovered, seed)
        cert = load_json(certificate)
        if (
            cert.get("status") != "PASS"
            or cert.get("experiment_id") != spec["experiment_id"]
            or cert.get("selected_checkpoint_sha256") != R07_CHECKPOINTS[seed]
            or cert.get("recovered_run_record_sha256") != R07_RUN_RECORDS[seed]["sha256"]
            or cert.get("scientific_training_reperformed") is not False
            or cert.get("protected_surface_opened") is not False
        ):
            raise TrackBOpsError(f"R07 {seed} terminal recovery certificate mismatch")

        view = source_views / f"r07_{seed.lower()}"
        view.mkdir(parents=True, exist_ok=False)
        _symlink_file(selected_checkpoint, view / "selected_checkpoint.pt")
        shutil.copy2(recovered, view / "run_record.json")
        views[f"r07_{seed.lower()}"] = view
        evidence[seed] = {
            "record_kind": "CRYPTOGRAPHIC_TERMINAL_RECORD_RECOVERY",
            "run_record_sha256": sha256_file(view / "run_record.json"),
            "run_id": R07_RUN_RECORDS[seed]["run_id"],
            "checkpoint_sha256": sha256_file(selected_checkpoint),
            "checkpoint_index_root": str(checkpoint_root),
            "restored_from_durable_locator": restored_from_durable,
            "recovery_certificate_sha256": sha256_file(certificate),
        }

    return views, evidence


def _prequalify_dino(root: Path, *, repo_root: Path) -> dict[str, object]:
    checkpoint = _find_exact_file_by_sha(
        root,
        expected_sha256=DINO_AUDIT_SHA256,
        suffixes={".pt", ".pth", ".ckpt", ".bin", ".safetensors", ""},
    )
    factory = _find_exact_file_by_sha(
        root,
        expected_sha256=DINO_FACTORY_MANIFEST_SHA256,
        suffixes={".json"},
        max_bytes=16 * 1024 * 1024,
    )
    factory_manifest = load_json(factory)
    factory_source_root = (repo_root / "journal_extension" / "teacher_factory").resolve()
    if not factory_source_root.is_dir():
        raise TrackBOpsError(f"frozen DINO teacher factory source root missing: {factory_source_root}")
    entrypoint = str(factory_manifest.get("entrypoint", "")).strip()
    if not entrypoint or ":" not in entrypoint:
        raise TrackBOpsError("DINO factory manifest lacks a valid entrypoint")
    errors = validate_teacher_factory_bundle(
        factory_manifest,
        source_root=factory_source_root,
        expected_entrypoint=entrypoint,
    )
    if errors:
        raise TrackBOpsError(
            "DINO teacher factory source qualification failed: " + "; ".join(errors)
        )

    try:
        teacher, teacher_identity = load_exact_teacher(
            checkpoint,
            factory_spec=entrypoint,
            factory_bundle_manifest=factory,
            repo_root=factory_source_root,
            factory_source_root=factory_source_root,
        )
    except Exception as exc:
        raise TrackBOpsError(
            f"DINO teacher preflight instantiation failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        parameter_count = sum(int(p.numel()) for p in teacher.parameters())
    finally:
        del teacher
    if parameter_count <= 0:
        raise TrackBOpsError("DINO teacher preflight produced a model with no parameters")

    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_source_path": str(checkpoint),
        "factory_manifest_sha256": sha256_file(factory),
        "factory_manifest_source_path": str(factory),
        "factory_source_root": str(factory_source_root),
        "factory_entrypoint": entrypoint,
        "factory_source_validation": "PASS",
        "teacher_instantiation": "PASS",
        "teacher_parameter_count": parameter_count,
        "teacher_bundle_sha256": str(teacher_identity.get("bundle_sha256", "")),
    }


def _write_source_qualification(
    output_root: Path,
    *,
    final_v1: tuple[Path, Path, Path],
    r07: dict[str, object],
    dino: dict[str, object],
    external_probe: dict[str, object],
    kaggle_owner: str | None,
) -> Path:
    manifest, class_map, image_root = final_v1
    payload = {
        "schema_version": "1.0",
        "status": "PASS_TRACKB_SOURCE_QUALIFICATION",
        "protected_external_prediction_count": 0,
        "v1_test_accessed": False,
        "final_v1": {
            "manifest_sha256": sha256_file(manifest),
            "class_map_sha256": sha256_file(class_map),
            "image_root": str(image_root),
            "train_present": (image_root / "train").is_dir(),
            "val_present": (image_root / "val").is_dir(),
        },
        "r07": r07,
        "dino": dino,
        "external_source_probe": external_probe,
        "kaggle_publication_owner": kaggle_owner,
    }
    path = output_root / "TRACKB_SOURCE_QUALIFICATION.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _resolve_core_file(core_root: Path, manifest: dict, key: str) -> Path:
    row = (manifest.get("files") or {}).get(key)
    if not isinstance(row, dict):
        raise TrackBOpsError(f"core bundle lacks required file key: {key}")
    path = (core_root / str(row.get("path", ""))).resolve()
    if core_root not in path.parents and path != core_root:
        raise TrackBOpsError(f"core file escapes bundle root: {key}")
    if not path.is_file():
        raise TrackBOpsError(f"core file missing: {key}: {path}")
    expected = str(row.get("sha256", ""))
    if len(expected) != 64 or sha256_file(path) != expected:
        raise TrackBOpsError(f"core file SHA mismatch: {key}")
    return path


def _run_core_builder(
    repo_root: Path,
    mounts: dict[str, Path],
    core_root: Path,
    final_v1_view_root: Path,
    r07_views: dict[str, Path],
) -> None:
    run_checked(
        [
            sys.executable,
            str(repo_root / "journal_extension/scripts/build_trackb_core_package.py"),
            "--repo-root", str(repo_root),
            "--final-v1-root", str(final_v1_view_root),
            "--r07-s1-root", str(r07_views["r07_s1"]),
            "--r07-s2-root", str(r07_views["r07_s2"]),
            "--r07-s3-root", str(r07_views["r07_s3"]),
            "--dino-bundle-root", str(mounts["dino_bundle"]),
            "--output-root", str(core_root),
        ],
        cwd=repo_root,
        timeout=7200,
    )


def _run_historical_builder(
    repo_root: Path,
    final_v1_root: Path,
    core_root: Path,
    historical_root: Path,
    device: str,
) -> None:
    v1_manifest, _class_map, image_root = _find_v1(final_v1_root)
    core = load_json(core_root / "TRACKB_INPUT_MANIFEST.json")
    dino = _resolve_core_file(core_root, core, "dino_checkpoint")
    factory = _resolve_core_file(core_root, core, "dino_factory_manifest")
    execution_lock = _resolve_core_file(core_root, core, "execution_lock")
    code_attestation = _resolve_core_file(core_root, core, "code_attestation")
    factory_source_root = (core_root / str(core["dino_factory_source_root"])).resolve()
    run_checked(
        [
            sys.executable,
            str(repo_root / "journal_extension/scripts/build_trackb_historical_compare.py"),
            "--v1-manifest", str(v1_manifest),
            "--image-root", str(image_root),
            "--dino-checkpoint", str(dino),
            "--factory-manifest", str(factory),
            "--factory-source-root", str(factory_source_root),
            "--repo-root", str(core_root / str(core["repository_root"])),
            "--execution-lock", str(execution_lock),
            "--code-attestation", str(code_attestation),
            "--output-dir", str(historical_root),
            "--device", device,
            "--workers", "4",
            "--orb-chunk-size", "512",
            "--dino-batch-size", "64",
        ],
        cwd=repo_root,
        timeout=21600,
    )


def _prepare_candidate(repo_root: Path, role: str, package_root: Path) -> None:
    run_checked(
        [
            sys.executable,
            str(repo_root / "journal_extension/scripts/prepare_trackb_candidate_input.py"),
            "--role", role,
            "--package-root", str(package_root),
            "--data-root", "data",
            "--source-metadata-record", "SOURCE_METADATA.json",
        ],
        cwd=repo_root,
        timeout=3600,
    )


def _verify_published_archive_roundtrip(
    slug: str,
    expected_manifest: dict,
    *,
    scratch_root: Path,
    attempts: int = 3,
) -> dict:
    expected_rows = expected_manifest.get("files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise TrackBOpsError("Kaggle content manifest has no files for archive verification")
    expected_payload_bytes = sum(int(row["bytes"]) for row in expected_rows)

    errors: list[str] = []
    for attempt in range(1, int(attempts) + 1):
        roundtrip_root = scratch_root / f"_roundtrip_{slug.split('/', 1)[-1]}_{attempt}"
        if roundtrip_root.exists():
            shutil.rmtree(roundtrip_root)
        roundtrip_root.mkdir(parents=True, exist_ok=False)
        try:
            free_bytes = int(shutil.disk_usage(roundtrip_root).free)
            hard_required = int(expected_payload_bytes * 1.05) + 256 * 1024 * 1024
            if free_bytes < hard_required:
                raise TrackBOpsError(
                    f"insufficient disk for Kaggle archive round-trip: slug={slug}, "
                    f"free={free_bytes}, hard_required={hard_required}, "
                    f"payload_bytes={expected_payload_bytes}"
                )
            run_checked(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(roundtrip_root), "-q"],
                timeout=7200,
            )
            archives = sorted(roundtrip_root.glob("*.zip"))
            if len(archives) != 1:
                raise TrackBOpsError(
                    f"expected one Kaggle round-trip ZIP for {slug}; found {archives}"
                )
            archive = archives[0]
            verified = 0
            with zipfile.ZipFile(archive) as zf:
                info_by_name = {
                    info.filename.replace("\\", "/").lstrip("./"): info
                    for info in zf.infolist()
                    if not info.is_dir()
                }
                manifest_names = {
                    str(row["path"]).replace("\\", "/").lstrip("./")
                    for row in expected_rows
                }
                unexpected_payload = sorted(
                    name for name in info_by_name
                    if name not in manifest_names
                    and name not in {"dataset-metadata.json", "TRACKB_KAGGLE_CONTENT_MANIFEST.json"}
                )
                if unexpected_payload:
                    raise TrackBOpsError(
                        f"Kaggle round-trip archive has unexpected payload members for {slug}: "
                        f"{unexpected_payload[:20]}"
                    )
                for row in expected_rows:
                    rel = str(row["path"]).replace("\\", "/").lstrip("./")
                    info = info_by_name.get(rel)
                    if info is None:
                        raise TrackBOpsError(
                            f"Kaggle round-trip archive missing member: {slug}/{rel}"
                        )
                    if int(info.file_size) != int(row["bytes"]):
                        raise TrackBOpsError(
                            f"Kaggle round-trip size mismatch: {slug}/{rel}"
                        )
                    h = hashlib.sha256()
                    with zf.open(info, "r") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            h.update(chunk)
                    if h.hexdigest() != str(row["sha256"]):
                        raise TrackBOpsError(
                            f"Kaggle round-trip SHA mismatch: {slug}/{rel}"
                        )
                    verified += 1
            result = {
                "status": "PASS",
                "archive_sha256": sha256_file(archive),
                "verified_file_count": verified,
                "content_digest_sha256": expected_manifest["content_digest_sha256"],
                "attempts": attempt,
                "peak_disk_guard_payload_bytes": expected_payload_bytes,
            }
            shutil.rmtree(roundtrip_root, ignore_errors=True)
            return result
        except Exception as exc:
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
            shutil.rmtree(roundtrip_root, ignore_errors=True)
            if attempt < int(attempts):
                continue
    raise TrackBOpsError(
        f"Kaggle archive round-trip failed after {attempts} attempts for {slug}: "
        + " | ".join(errors[-3:])
    )



def _manifest_sha(root: Path) -> str:
    path = root / "TRACKB_INPUT_MANIFEST.json"
    if not path.is_file():
        raise TrackBOpsError(f"Track-B input manifest missing: {path}")
    return sha256_file(path)


def _role_content_identity(root: Path) -> dict[str, object]:
    """Cryptographically bind every regular payload file inside a Track-B role root.

    Pair receipts live above the role roots, so this ledger cannot hash itself.
    Symlinks are forbidden in published role payloads because they are not portable
    across Kaggle materialization boundaries.
    """
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TrackBOpsError(f"published Track-B role contains a symlink: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rows.append({
            "path": rel,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    if not rows:
        raise TrackBOpsError(f"Track-B role payload is empty: {root}")
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "content_sha256": sha256_json(rows),
    }


def _validate_roles(
    core_root: Path,
    historical_root: Path,
    gvlid_root: Path,
    potato_root: Path,
) -> None:
    expected = {
        core_root: "core",
        historical_root: "historical_compare",
        gvlid_root: "gvlid_v5",
        potato_root: "irish_potato",
    }
    for root, role in expected.items():
        obj = load_json(root / "TRACKB_INPUT_MANIFEST.json")
        if obj.get("role") != role:
            raise TrackBOpsError(f"input-role mismatch for {root}: expected={role}, got={obj.get('role')}")
    hist = load_json(historical_root / "TRACKB_INPUT_MANIFEST.json")
    required_hist = {
        "coverage_scope": "V1_TRAIN_VAL_ONLY",
        "image_count": 92744,
        "maximum_evidence_grade": "EXT-S",
        "v1_test_image_bytes_accessed": False,
    }
    for key, value in required_hist.items():
        if hist.get(key) != value:
            raise TrackBOpsError(f"historical comparison contract mismatch: {key}={hist.get(key)!r}")


def _write_pair_receipts(
    *,
    repo_root: Path,
    infra_root: Path,
    external_root: Path,
) -> dict:
    core_root = infra_root / "core"
    historical_root = infra_root / "historical_compare"
    gvlid_root = external_root / "gvlid_v5"
    potato_root = external_root / "irish_potato"
    core = load_json(core_root / "TRACKB_INPUT_MANIFEST.json")
    execution_lock = _resolve_core_file(core_root, core, "execution_lock")
    code_attestation = _resolve_core_file(core_root, core, "code_attestation")
    source_sha = run_checked(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], timeout=120
    ).stdout.strip()
    if len(source_sha) != 40:
        raise TrackBOpsError("materialization requires an exact Git source SHA")

    role_manifest_sha256 = {
        "core": _manifest_sha(core_root),
        "historical_compare": _manifest_sha(historical_root),
        "gvlid_v5": _manifest_sha(gvlid_root),
        "irish_potato": _manifest_sha(potato_root),
    }
    role_content_identity = {
        "core": _role_content_identity(core_root),
        "historical_compare": _role_content_identity(historical_root),
        "gvlid_v5": _role_content_identity(gvlid_root),
        "irish_potato": _role_content_identity(potato_root),
    }
    pairing_preimage = {
        "schema_version": "2.0",
        "repository_source_sha": source_sha,
        "scientific_execution_lock_sha256": sha256_file(execution_lock),
        "scientific_code_attestation_sha256": sha256_file(code_attestation),
        "role_manifest_sha256": role_manifest_sha256,
        "role_content_identity": role_content_identity,
    }
    materialization_id = sha256_json(pairing_preimage)

    common = {
        "schema_version": "2.0",
        "status": "PASS_PAIRED_TRACKB_INPUT_BUNDLE",
        "materialization_id": materialization_id,
        **pairing_preimage,
    }
    infra_receipt = {
        **common,
        "bundle_role": "TRACKB_INFRASTRUCTURE",
        "contained_roles": ["core", "historical_compare"],
        "counterpart_bundle_role": "TRACKB_EXTERNAL",
    }
    external_receipt = {
        **common,
        "bundle_role": "TRACKB_EXTERNAL",
        "contained_roles": ["gvlid_v5", "irish_potato"],
        "counterpart_bundle_role": "TRACKB_INFRASTRUCTURE",
    }
    (infra_root / "TRACKB_INFRASTRUCTURE_BUNDLE.json").write_text(
        json.dumps(infra_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (external_root / "TRACKB_EXTERNAL_BUNDLE.json").write_text(
        json.dumps(external_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "materialization_id": materialization_id,
        "repository_source_sha": source_sha,
        "role_manifest_sha256": role_manifest_sha256,
        "role_content_identity": role_content_identity,
        "scientific_execution_lock_sha256": pairing_preimage["scientific_execution_lock_sha256"],
        "scientific_code_attestation_sha256": pairing_preimage["scientific_code_attestation_sha256"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CropCop Track-B v4 readiness/materialization controller. No protected external predictions."
    )
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--output-root", default="/kaggle/working/trackb_v4_materialization")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kaggle-owner", default="AUTO")
    ap.add_argument("--skip-publication", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    stage("0 :: attached-input discovery")
    mounts = {
        role: _resolve_attached_root(input_root, basename)
        for role, basename in SOURCE_SLUG_BASENAMES.items()
    }
    print(json.dumps({key: str(value) for key, value in mounts.items()}, indent=2, sort_keys=True))

    stage("0.5 :: complete source qualification")
    early_owner = None
    if not args.skip_publication:
        configure_runtime_secrets(require_github=False)
        early_owner = verify_authenticated_kaggle_owner(args.kaggle_owner)
    external_probe = probe_external_sources()
    if external_probe.get("status") != "PASS":
        raise TrackBOpsError(f"external-source readiness probe did not PASS: {external_probe}")

    source_views = output_root / "_source_views"
    final_v1_authority = _find_v1(mounts["final_v1"])
    final_v1_view = _prepare_final_v1_core_view(
        mounts["final_v1"],
        source_views / "final_v1",
    )
    r07_views, r07_evidence = _prepare_r07_source_views(repo_root, mounts, source_views)
    dino_evidence = _prequalify_dino(mounts["dino_bundle"], repo_root=repo_root)
    source_qualification_path = _write_source_qualification(
        output_root,
        final_v1=final_v1_authority,
        r07=r07_evidence,
        dino=dino_evidence,
        external_probe=external_probe,
        kaggle_owner=early_owner,
    )
    print(source_qualification_path.read_text(encoding="utf-8"), flush=True)

    infra_root = output_root / "infrastructure_bundle"
    external_root = output_root / "external_bundle"
    core_root = infra_root / "core"
    historical_root = infra_root / "historical_compare"
    gvlid_root = external_root / "gvlid_v5"
    potato_root = external_root / "irish_potato"
    infra_root.mkdir(parents=True)
    external_root.mkdir(parents=True)

    stage("1 :: authoritative external cohorts — fail-fast sealed acquisition")
    lineage_review = repo_root / "journal_extension/track_b_r07/TRACKB_EXTERNAL_LINEAGE_REVIEW_v1.json"
    acquire_gvlid_v5(
        gvlid_root,
        lineage_review_path=lineage_review,
        expected_source_manifest_sha256=str(
            (external_probe.get("gvlid_v5") or {}).get("public_api_manifest_sha256", "")
        ),
    )
    _prepare_candidate(repo_root, "gvlid_v5", gvlid_root)
    acquire_irish_potato(
        potato_root,
        lineage_review_path=lineage_review,
        expected_source_manifest_sha256=str(
            (external_probe.get("irish_potato") or {}).get("source_manifest_sha256", "")
        ),
    )
    _prepare_candidate(repo_root, "irish_potato", potato_root)

    stage("2 :: immutable core")
    _run_core_builder(repo_root, mounts, core_root, final_v1_view, r07_views)

    stage("3 :: safe historical comparison")
    _run_historical_builder(repo_root, mounts["final_v1"], core_root, historical_root, args.device)
    shutil.rmtree(source_views, ignore_errors=True)

    stage("4 :: input-contract validation and pairing")
    _validate_roles(core_root, historical_root, gvlid_root, potato_root)
    pairing = _write_pair_receipts(
        repo_root=repo_root,
        infra_root=infra_root,
        external_root=external_root,
    )

    readiness = {
        "schema_version": "1.0",
        "status": "PASS_TRACKB_MATERIALIZATION_LOCAL",
        "protected_external_prediction_count": 0,
        "v1_test_accessed": False,
        "materialization": pairing,
        "source_qualification_sha256": sha256_file(source_qualification_path),
        "publication": None,
    }

    if not args.skip_publication:
        stage("5 :: private Kaggle publication and full round-trip verification")
        owner = early_owner or verify_authenticated_kaggle_owner(args.kaggle_owner)
        infra_slug = (
            f"{owner}/{INFRA_DATASET_PREFIX}-{pairing['materialization_id'][:16]}"
        )
        external_slug = (
            f"{owner}/{EXTERNAL_DATASET_PREFIX}-{pairing['materialization_id'][:16]}"
        )
        infra_pub = publish_private_kaggle_dataset(
            folder=infra_root,
            slug=infra_slug,
            title="CropCop Track B R07 Infrastructure v5",
            version_message=f"Track-B materialization {pairing['materialization_id'][:16]}",
            license_name="other",
            full_roundtrip=False,
            allow_version=False,
        )
        infra_manifest = load_json(infra_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")
        shutil.rmtree(infra_root, ignore_errors=True)
        infra_pub["archive_roundtrip"] = _verify_published_archive_roundtrip(
            infra_slug,
            infra_manifest,
            scratch_root=output_root,
        )

        external_pub = publish_private_kaggle_dataset(
            folder=external_root,
            slug=external_slug,
            title="CropCop Track B R07 External Cohorts v5",
            version_message=f"Track-B materialization {pairing['materialization_id'][:16]}",
            license_name="other",
            full_roundtrip=False,
            allow_version=False,
        )
        external_manifest = load_json(external_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")
        shutil.rmtree(external_root, ignore_errors=True)
        external_pub["archive_roundtrip"] = _verify_published_archive_roundtrip(
            external_slug,
            external_manifest,
            scratch_root=output_root,
        )

        readiness["publication"] = {
            "owner": owner,
            "infrastructure": infra_pub,
            "external": external_pub,
        }
        readiness["status"] = "PASS_TRACKB_INPUT_MATERIALIZATION"

    receipt_path = output_root / "TRACKB_READINESS_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(readiness, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
