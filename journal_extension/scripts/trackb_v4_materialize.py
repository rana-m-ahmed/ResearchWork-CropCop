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
from cropcop_je.trackb_r07 import CLASS_MAP_SHA256, DATASET_MANIFEST_SHA256, TrackBError, load_json
from cropcop_je.trackb_r07_ops import (
    TrackBOpsError,
    acquire_gvlid_v5,
    acquire_irish_potato,
    configure_runtime_secrets,
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

INFRA_DATASET_NAME = "cropcop-trackb-r07-infrastructure-v4"
EXTERNAL_DATASET_NAME = "cropcop-trackb-r07-external-v4"


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
) -> None:
    run_checked(
        [
            sys.executable,
            str(repo_root / "journal_extension/scripts/build_trackb_core_package.py"),
            "--repo-root", str(repo_root),
            "--final-v1-root", str(final_v1_view_root),
            "--r07-s1-root", str(mounts["r07_s1"]),
            "--r07-s2-root", str(mounts["r07_s2"]),
            "--r07-s3-root", str(mounts["r07_s3"]),
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


def _verify_published_archive_roundtrip(slug: str, published_folder: Path) -> dict:
    manifest_path = published_folder / "TRACKB_KAGGLE_CONTENT_MANIFEST.json"
    if not manifest_path.is_file():
        raise TrackBOpsError(f"local Kaggle content manifest missing after publication: {manifest_path}")
    manifest = load_json(manifest_path)
    expected_rows = manifest.get("files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise TrackBOpsError("local Kaggle content manifest has no files")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_checked(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(root), "-q"],
            timeout=7200,
        )
        archives = sorted(root.glob("*.zip"))
        if len(archives) != 1:
            raise TrackBOpsError(f"expected one Kaggle round-trip ZIP for {slug}; found {archives}")
        archive = archives[0]
        verified = 0
        with zipfile.ZipFile(archive) as zf:
            info_by_name = {
                info.filename.replace("\\", "/").lstrip("./"): info
                for info in zf.infolist()
                if not info.is_dir()
            }
            for row in expected_rows:
                rel = str(row["path"]).replace("\\", "/").lstrip("./")
                info = info_by_name.get(rel)
                if info is None:
                    raise TrackBOpsError(f"Kaggle round-trip archive missing member: {slug}/{rel}")
                if int(info.file_size) != int(row["bytes"]):
                    raise TrackBOpsError(f"Kaggle round-trip size mismatch: {slug}/{rel}")
                h = hashlib.sha256()
                with zf.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest() != str(row["sha256"]):
                    raise TrackBOpsError(f"Kaggle round-trip SHA mismatch: {slug}/{rel}")
                verified += 1
        return {
            "status": "PASS",
            "archive_sha256": sha256_file(archive),
            "verified_file_count": verified,
            "content_digest_sha256": manifest["content_digest_sha256"],
        }


def _manifest_sha(root: Path) -> str:
    path = root / "TRACKB_INPUT_MANIFEST.json"
    if not path.is_file():
        raise TrackBOpsError(f"Track-B input manifest missing: {path}")
    return sha256_file(path)


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
    pairing_preimage = {
        "schema_version": "1.0",
        "repository_source_sha": source_sha,
        "scientific_execution_lock_sha256": sha256_file(execution_lock),
        "scientific_code_attestation_sha256": sha256_file(code_attestation),
        "role_manifest_sha256": role_manifest_sha256,
    }
    materialization_id = sha256_json(pairing_preimage)

    common = {
        "schema_version": "1.0",
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

    stage("1 :: immutable core")
    infra_root = output_root / "infrastructure_bundle"
    external_root = output_root / "external_bundle"
    core_root = infra_root / "core"
    historical_root = infra_root / "historical_compare"
    gvlid_root = external_root / "gvlid_v5"
    potato_root = external_root / "irish_potato"
    infra_root.mkdir(parents=True)
    external_root.mkdir(parents=True)

    source_views = output_root / "_source_views"
    final_v1_view = _prepare_final_v1_core_view(
        mounts["final_v1"],
        source_views / "final_v1",
    )
    _run_core_builder(repo_root, mounts, core_root, final_v1_view)

    stage("2 :: safe historical comparison")
    _run_historical_builder(repo_root, mounts["final_v1"], core_root, historical_root, args.device)
    shutil.rmtree(source_views, ignore_errors=True)

    stage("3 :: authoritative external cohorts")
    lineage_review = repo_root / "journal_extension/track_b_r07/TRACKB_EXTERNAL_LINEAGE_REVIEW_v1.json"
    acquire_gvlid_v5(gvlid_root, lineage_review_path=lineage_review)
    _prepare_candidate(repo_root, "gvlid_v5", gvlid_root)
    acquire_irish_potato(potato_root, lineage_review_path=lineage_review)
    _prepare_candidate(repo_root, "irish_potato", potato_root)

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
        "publication": None,
    }

    if not args.skip_publication:
        stage("5 :: private Kaggle publication and full round-trip verification")
        configure_runtime_secrets(require_github=False)
        owner = verify_authenticated_kaggle_owner(args.kaggle_owner)
        infra_slug = f"{owner}/{INFRA_DATASET_NAME}"
        external_slug = f"{owner}/{EXTERNAL_DATASET_NAME}"
        infra_pub = publish_private_kaggle_dataset(
            folder=infra_root,
            slug=infra_slug,
            title="CropCop Track B R07 Infrastructure v4",
            version_message=f"Track-B materialization {pairing['materialization_id'][:16]}",
            license_name="other",
            full_roundtrip=False,
        )
        infra_pub["archive_roundtrip"] = _verify_published_archive_roundtrip(
            infra_slug, infra_root
        )
        shutil.rmtree(infra_root, ignore_errors=True)

        external_pub = publish_private_kaggle_dataset(
            folder=external_root,
            slug=external_slug,
            title="CropCop Track B R07 External Cohorts v4",
            version_message=f"Track-B materialization {pairing['materialization_id'][:16]}",
            license_name="other",
            full_roundtrip=False,
        )
        external_pub["archive_roundtrip"] = _verify_published_archive_roundtrip(
            external_slug, external_root
        )
        shutil.rmtree(external_root, ignore_errors=True)

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
