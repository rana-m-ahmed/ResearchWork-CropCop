from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import (
    CLASS_MAP_SHA256,
    DATASET_MANIFEST_SHA256,
    DINO_AUDIT_SHA256,
    DINO_FACTORY_MANIFEST_SHA256,
    R07_CHECKPOINTS,
    R07_RUN_RECORDS,
    TrackBError,
)

VAL_COUNT = 16368

EXPECTED_SOURCE_HINTS = {
    "final_v1": "ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1",
    "r07_s1": "sabahatabbas/sec-je-r07-cnxtt-context-s1-8904b100d223-a01",
    "r07_s2": "sabahatabbas/cropcop-r07-cnxtt-context-s2-abce1197-56023042",
    "r07_s3": "sabahatabbas/cropcop-r07-cnxtt-context-s3-f13ca687-56023042",
    "dino_bundle": "ranamuhammadahmed6/cropcop-secondary-g1-8904b100@version-2",
}


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _find_exact_sha(root: Path, expected_sha: str, *, kind: str) -> Path:
    candidates: list[Path] = []
    if kind == "model":
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_size < 10 * 1024 * 1024:
                    continue
            except OSError:
                continue
            if path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ""}:
                continue
            candidates.append(path)
    elif kind == "json":
        candidates = [
            p for p in root.rglob("*.json")
            if p.is_file() and p.stat().st_size <= 16 * 1024 * 1024
        ]
    else:
        raise ValueError(kind)

    matches = [path for path in sorted(candidates) if sha256_file(path) == expected_sha]
    if len(matches) != 1:
        raise TrackBError(
            f"expected exactly one {kind} file with SHA-256 {expected_sha} under {root}; "
            f"found {[str(p) for p in matches]}"
        )
    return matches[0]


def _find_final_v1(root: Path) -> tuple[Path, Path, Path]:
    manifest_matches = [
        p.resolve() for p in root.rglob("final_manifest.csv")
        if p.is_file() and sha256_file(p) == DATASET_MANIFEST_SHA256
    ]
    class_matches = [
        p.resolve() for p in root.rglob("class_to_idx.json")
        if p.is_file() and sha256_file(p) == CLASS_MAP_SHA256
    ]
    if len(manifest_matches) != 1 or len(class_matches) != 1:
        raise TrackBError(
            f"Final-V1 authority discovery failed: manifest={manifest_matches}, class_map={class_matches}"
        )
    manifest = manifest_matches[0]
    class_map = class_matches[0]
    candidate_roots = []
    for parent in [manifest.parent.parent, manifest.parent, manifest.parent.parent.parent]:
        image_root = parent / "dataset"
        if (image_root / "val").is_dir():
            candidate_roots.append(image_root.resolve())
    candidate_roots = list(dict.fromkeys(candidate_roots))
    if len(candidate_roots) != 1:
        raise TrackBError(f"could not resolve unique Final-V1 dataset root from {manifest}")
    return manifest, class_map, candidate_roots[0]


def _copy_validation_surface(manifest: Path, image_root: Path, output_root: Path) -> dict:
    val_rows: list[tuple[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"record_key", "portable_relpath", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise TrackBError(f"Final-V1 manifest missing columns: {sorted(missing)}")
        for row in reader:
            if str(row["split"]).strip() != "val":
                continue
            row_id = str(row["record_key"]).strip()
            rel = str(row["portable_relpath"]).strip().replace("\\", "/")
            posix = PurePosixPath(rel)
            if not row_id or posix.is_absolute() or ".." in posix.parts or not rel.startswith("val/"):
                raise TrackBError(f"unsafe validation row: row_id={row_id!r}, path={rel!r}")
            val_rows.append((row_id, rel))
    if len(val_rows) != VAL_COUNT:
        raise TrackBError(f"Final-V1 validation count mismatch: expected {VAL_COUNT}, got {len(val_rows)}")
    if len({row_id for row_id, _ in val_rows}) != VAL_COUNT or len({rel for _, rel in val_rows}) != VAL_COUNT:
        raise TrackBError("Final-V1 validation surface has duplicate row IDs or paths")

    destination = output_root / "v1_validation"
    copied = 0
    copied_bytes = 0
    for row_id, rel in val_rows:
        src = (image_root / rel).resolve()
        if image_root not in src.parents:
            raise TrackBError(f"validation source path escapes image root: {rel}")
        if not src.is_file():
            raise FileNotFoundError(f"validation image missing: {src}")
        dst = destination / rel
        _copy_file(src, dst)
        copied += 1
        copied_bytes += dst.stat().st_size
        if copied % 2000 == 0:
            print(f"Validation copy {copied}/{VAL_COUNT}", flush=True)

    if (destination / "test").exists():
        raise TrackBError("core builder copied a forbidden test directory")
    return {"rows": copied, "bytes": copied_bytes, "root": destination}


def _verify_run_record(path: Path, seed: str) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if str(obj.get("run_id", "")) != R07_RUN_RECORDS[seed]["run_id"]:
        raise TrackBError(f"R07 {seed} run-record ID mismatch")
    selected_sha = (
        ((obj.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
        or (obj.get("result_summary") or {}).get("selected_checkpoint_sha256")
        or obj.get("selected_checkpoint_sha256")
    )
    if str(selected_sha) != R07_CHECKPOINTS[seed]:
        raise TrackBError(f"R07 {seed} run record does not bind selected checkpoint")
    metrics = (obj.get("result_summary") or {}).get("selected_metrics") or {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "nll"):
        if key not in metrics:
            raise TrackBError(f"R07 {seed} run record lacks selected metric {key}")
    return obj


def _copy_repository(source_repo: Path, dest_repo: Path) -> str | None:
    if not (source_repo / "journal_extension").is_dir():
        raise TrackBError(f"repository root missing journal_extension/: {source_repo}")
    dest_repo.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source_repo / "journal_extension",
        dest_repo / "journal_extension",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    try:
        cp = subprocess.run(
            ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return cp.stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assemble the immutable CropCop Track-B core Kaggle package from authoritative mounted inputs."
    )
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--final-v1-root", required=True)
    ap.add_argument("--r07-s1-root", required=True)
    ap.add_argument("--r07-s2-root", required=True)
    ap.add_argument("--r07-s3-root", required=True)
    ap.add_argument("--dino-bundle-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    final_v1_root = Path(args.final_v1_root).resolve()
    r07_roots = {
        "S1": Path(args.r07_s1_root).resolve(),
        "S2": Path(args.r07_s2_root).resolve(),
        "S3": Path(args.r07_s3_root).resolve(),
    }
    dino_root = Path(args.dino_bundle_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrackBError(f"core output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    discovered: dict[str, object] = {
        "schema_version": "1.0",
        "source_hints": EXPECTED_SOURCE_HINTS,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "consumed_test_image_bytes_copied": False,
    }

    repo_copy = output / "repository"
    repo_head = _copy_repository(repo_root, repo_copy)
    discovered["repository_source_head"] = repo_head

    manifest, class_map, image_root = _find_final_v1(final_v1_root)
    _copy_file(manifest, output / "authority" / "v1" / "final_manifest.csv")
    _copy_file(class_map, output / "authority" / "v1" / "class_to_idx.json")
    val = _copy_validation_surface(manifest, image_root, output)
    discovered["validation_surface"] = {"rows": val["rows"], "bytes": val["bytes"]}

    model_dir = output / "models"
    for seed, root in r07_roots.items():
        checkpoint = _find_exact_sha(root, R07_CHECKPOINTS[seed], kind="model")
        run_record = _find_exact_sha(root, R07_RUN_RECORDS[seed]["sha256"], kind="json")
        _verify_run_record(run_record, seed)
        checkpoint_dest = model_dir / f"r07_{seed.lower()}.pt"
        record_dest = model_dir / f"r07_{seed.lower()}_run_record.json"
        _copy_file(checkpoint, checkpoint_dest)
        _copy_file(run_record, record_dest)
        discovered[f"r07_{seed.lower()}"] = {
            "checkpoint_source_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint_dest),
            "run_record_source_path": str(run_record),
            "run_record_sha256": sha256_file(record_dest),
            "run_id": R07_RUN_RECORDS[seed]["run_id"],
        }

    dino_checkpoint = _find_exact_sha(dino_root, DINO_AUDIT_SHA256, kind="model")
    dino_factory = _find_exact_sha(dino_root, DINO_FACTORY_MANIFEST_SHA256, kind="json")
    dino_dir = output / "audit_encoder"
    _copy_file(dino_checkpoint, dino_dir / "DINO_TEACHER.pt")
    _copy_file(dino_factory, dino_dir / "TEACHER_FACTORY_BUNDLE.json")
    discovered["dino"] = {
        "checkpoint_source_path": str(dino_checkpoint),
        "checkpoint_sha256": sha256_file(dino_dir / "DINO_TEACHER.pt"),
        "factory_source_path": str(dino_factory),
        "factory_sha256": sha256_file(dino_dir / "TEACHER_FACTORY_BUNDLE.json"),
    }

    locator_path = output / "SOURCE_LOCATORS.json"
    atomic_write_json(locator_path, discovered)

    prep = repo_copy / "journal_extension" / "scripts" / "prepare_trackb_core_input.py"
    cmd = [
        sys.executable, str(prep),
        "--package-root", str(output),
        "--repository-root", "repository",
        "--v1-validation-root", "v1_validation",
        "--downstream-authority", "repository/journal_extension/amendments/track_bc_r07_downstream_v2.json",
        "--execution-lock", "repository/journal_extension/track_b_r07/TRACKB_R07_EXECUTION_LOCK_v2.json",
        "--code-attestation", "repository/journal_extension/track_b_r07/TRACKB_CODE_ATTESTATION_v2.json",
        "--class-map", "authority/v1/class_to_idx.json",
        "--v1-manifest", "authority/v1/final_manifest.csv",
        "--r07-s1", "models/r07_s1.pt",
        "--r07-s1-run-record", "models/r07_s1_run_record.json",
        "--r07-s2", "models/r07_s2.pt",
        "--r07-s2-run-record", "models/r07_s2_run_record.json",
        "--r07-s3", "models/r07_s3.pt",
        "--r07-s3-run-record", "models/r07_s3_run_record.json",
        "--dino-checkpoint", "audit_encoder/DINO_TEACHER.pt",
        "--dino-factory-manifest", "audit_encoder/TEACHER_FACTORY_BUNDLE.json",
        "--dino-factory-source-root", "repository",
    ]
    subprocess.run(cmd, cwd=repo_copy, check=True)

    input_manifest = json.loads((output / "TRACKB_INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    if input_manifest.get("role") != "core":
        raise TrackBError("core input manifest role mismatch after preparation")
    if any(
        marker in json.dumps(input_manifest, sort_keys=True).lower()
        for marker in ("v1_test", "test_consumed", "ds-v1-test")
    ):
        raise TrackBError("forbidden consumed-test marker in final core input manifest")

    certificate = {
        "schema_version": "1.0",
        "status": "PASS",
        "kind": "trackb_core_package_build",
        "repository_source_head": repo_head,
        "validation_rows": VAL_COUNT,
        "consumed_test_image_bytes_copied": False,
        "r07_checkpoint_sha256": R07_CHECKPOINTS,
        "r07_run_record_sha256": {seed: R07_RUN_RECORDS[seed]["sha256"] for seed in ("S1", "S2", "S3")},
        "dino_checkpoint_sha256": DINO_AUDIT_SHA256,
        "dino_factory_manifest_sha256": DINO_FACTORY_MANIFEST_SHA256,
        "source_locator_record_sha256": sha256_file(locator_path),
        "trackb_input_manifest_sha256": sha256_file(output / "TRACKB_INPUT_MANIFEST.json"),
    }
    atomic_write_json(output / "CORE_BUILD_CERTIFICATE.json", certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
