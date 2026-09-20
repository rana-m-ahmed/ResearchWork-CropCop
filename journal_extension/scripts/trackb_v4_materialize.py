from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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


def _unique_sha(root: Path, name: str, expected_sha: str) -> Path:
    matches = [
        path.resolve()
        for path in root.rglob(name)
        if path.is_file() and sha256_file(path) == expected_sha
    ]
    if len(matches) != 1:
        raise TrackBOpsError(
            f"expected one {name} with SHA-256 {expected_sha} under {root}; found {matches}"
        )
    return matches[0]


def _find_v1(root: Path) -> tuple[Path, Path, Path]:
    manifest = _unique_sha(root, "final_manifest.csv", DATASET_MANIFEST_SHA256)
    class_map = _unique_sha(root, "class_to_idx.json", CLASS_MAP_SHA256)
    roots = []
    for parent in (manifest.parent.parent, manifest.parent, manifest.parent.parent.parent):
        image_root = parent / "dataset"
        if (image_root / "train").is_dir() and (image_root / "val").is_dir():
            roots.append(image_root.resolve())
    roots = list(dict.fromkeys(roots))
    if len(roots) != 1:
        raise TrackBOpsError(f"could not resolve unique frozen Final-V1 image root: {roots}")
    return manifest, class_map, roots[0]


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


def _run_core_builder(repo_root: Path, mounts: dict[str, Path], core_root: Path) -> None:
    run_checked(
        [
            sys.executable,
            str(repo_root / "journal_extension/scripts/build_trackb_core_package.py"),
            "--repo-root", str(repo_root),
            "--final-v1-root", str(mounts["final_v1"]),
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
    _run_core_builder(repo_root, mounts, core_root)

    stage("2 :: safe historical comparison")
    _run_historical_builder(repo_root, mounts["final_v1"], core_root, historical_root, args.device)

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
            full_roundtrip=True,
        )
        external_pub = publish_private_kaggle_dataset(
            folder=external_root,
            slug=external_slug,
            title="CropCop Track B R07 External Cohorts v4",
            version_message=f"Track-B materialization {pairing['materialization_id'][:16]}",
            license_name="other",
            full_roundtrip=True,
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
