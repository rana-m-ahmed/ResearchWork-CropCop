from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import (
    CLASS_MAP_SHA256,
    DATASET_MANIFEST_SHA256,
    DINO_AUDIT_SHA256,
    R07_CHECKPOINTS,
    TrackBError,
    load_json,
    validate_downstream_authority,
    validate_execution_lock,
    verify_code_attestation,
)


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root not in path.parents and path != root:
        raise TrackBError(f"path escapes package root: {value}")
    return path


def _file(root: Path, value: str, label: str) -> dict:
    path = _inside(root, value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description="Create and validate the immutable Track-B core Kaggle input manifest.")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--repository-root", required=True, help="Path relative to package root containing the frozen repository snapshot.")
    ap.add_argument("--v1-validation-root", required=True, help="Path relative to package root ABOVE the val/ subtree; no V1 test bytes may be present/used.")
    ap.add_argument("--downstream-authority", required=True)
    ap.add_argument("--execution-lock", required=True)
    ap.add_argument("--code-attestation", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--v1-manifest", required=True)
    for seed in ("s1", "s2", "s3"):
        ap.add_argument(f"--r07-{seed}", required=True)
        ap.add_argument(f"--r07-{seed}-run-record", required=True)
    ap.add_argument("--dino-checkpoint", required=True)
    ap.add_argument("--dino-factory-manifest", required=True)
    ap.add_argument("--dino-factory-source-root", required=True)
    args = ap.parse_args()

    root = Path(args.package_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = _inside(root, args.repository_root)
    val_root = _inside(root, args.v1_validation_root)
    dino_source = _inside(root, args.dino_factory_source_root)
    if not repo.is_dir() or not val_root.is_dir() or not dino_source.is_dir():
        raise TrackBError("repository, validation, and DINO source roots must be directories inside package root")

    file_args = {
        "downstream_authority": args.downstream_authority,
        "execution_lock": args.execution_lock,
        "code_attestation": args.code_attestation,
        "class_map": args.class_map,
        "v1_manifest": args.v1_manifest,
        "r07_s1": args.r07_s1,
        "r07_s2": args.r07_s2,
        "r07_s3": args.r07_s3,
        "r07_s1_run_record": args.r07_s1_run_record,
        "r07_s2_run_record": args.r07_s2_run_record,
        "r07_s3_run_record": args.r07_s3_run_record,
        "dino_checkpoint": args.dino_checkpoint,
        "dino_factory_manifest": args.dino_factory_manifest,
    }
    files = {key: _file(root, value, key) for key, value in file_args.items()}
    if files["class_map"]["sha256"] != CLASS_MAP_SHA256:
        raise TrackBError("class-map SHA mismatch")
    if files["v1_manifest"]["sha256"] != DATASET_MANIFEST_SHA256:
        raise TrackBError("V1 manifest SHA mismatch")
    for seed in ("S1", "S2", "S3"):
        if files[f"r07_{seed.lower()}"]["sha256"] != R07_CHECKPOINTS[seed]:
            raise TrackBError(f"R07 {seed} checkpoint SHA mismatch")
    if files["dino_checkpoint"]["sha256"] != DINO_AUDIT_SHA256:
        raise TrackBError("DINO audit checkpoint SHA mismatch")

    validate_downstream_authority(load_json(_inside(root, args.downstream_authority)))
    lock = load_json(_inside(root, args.execution_lock))
    validate_execution_lock(lock)
    attestation_path = _inside(root, args.code_attestation)
    if files["code_attestation"]["sha256"] != str(lock.get("code_attestation_sha256", "")):
        raise TrackBError("code-attestation SHA differs from Track-B execution lock")
    verify_code_attestation(repo, attestation_path)

    # Fail fast on the most common packaging mistake: pointing at val/ instead of its parent.
    if val_root.name.lower() == "val":
        raise TrackBError("--v1-validation-root must be the directory ABOVE val/ because frozen manifest paths already start with 'val/'")
    if not (val_root / "val").is_dir():
        raise TrackBError("frozen V1 validation package must expose a val/ subtree under --v1-validation-root")
    forbidden_dirs = [p for p in val_root.iterdir() if p.is_dir() and p.name.lower() in {"test", "test_consumed", "v1_test", "ds-v1-test-consumed"}]
    if forbidden_dirs:
        raise TrackBError("core package contains consumed V1-test directory beside val/: " + ", ".join(p.name for p in forbidden_dirs))

    manifest = {
        "schema_version": "1.0",
        "role": "core",
        "repository_root": repo.relative_to(root).as_posix(),
        "v1_validation_root": val_root.relative_to(root).as_posix(),
        "dino_factory_source_root": dino_source.relative_to(root).as_posix(),
        "files": files,
    }
    text = json.dumps(manifest, sort_keys=True).lower()
    for marker in ("ds-v1-test-consumed", "v1_test", "test_consumed"):
        if marker in text:
            raise TrackBError(f"forbidden consumed-test marker in core package manifest: {marker}")
    atomic_write_json(root / "TRACKB_INPUT_MANIFEST.json", manifest)
    print(json.dumps({"status":"PASS","role":"core","manifest":str(root / 'TRACKB_INPUT_MANIFEST.json')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
