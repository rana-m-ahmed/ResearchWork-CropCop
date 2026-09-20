from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import (
    CLASS_MAP_SHA256,
    DATASET_MANIFEST_SHA256,
    TrackBError,
    load_json,
)
from cropcop_je.trackb_r07_ops import (
    KAGGLE_OWNER_DEFAULT,
    SOURCE_DATASETS,
    evidence_dataset_slug,
    historical_dataset_slug,
    TrackBOpsError,
    acquire_gvlid_v5,
    acquire_irish_potato,
    configure_runtime_secrets,
    download_kaggle_dataset,
    ensure_kaggle_cli,
    kaggle_dataset_exists,
    prepare_private_evidence_folder,
    publish_private_kaggle_dataset,
    publish_public_trackb_evidence,
    probe_external_sources,
    redact,
    run_checked,
    utc_now,
    verify_authenticated_kaggle_owner,
    verify_github_repository_push_access,
    verify_kaggle_source_access,
)


def stage(name: str) -> None:
    print(f"\n{'=' * 18} {name} {'=' * 18}", flush=True)


def disk_gb(path: Path) -> dict[str, float]:
    usage = shutil.disk_usage(path)
    return {
        "total": round(usage.total / 1024**3, 2),
        "used": round(usage.used / 1024**3, 2),
        "free": round(usage.free / 1024**3, 2),
    }


def assert_persistent_output_hygiene(workspace: Path, output_root: Path) -> dict[str, int]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    model_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".pte"}
    forbidden = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in image_suffixes or suffix in model_suffixes:
            forbidden.append(path)
        if path.name.endswith(".part"):
            forbidden.append(path)
    if forbidden:
        raise TrackBOpsError(
            "persistent Track-B workspace contains forbidden raw/model/partial artifacts: "
            + ", ".join(str(p.relative_to(workspace)) for p in forbidden[:20])
        )
    return {
        "persistent_file_count": sum(1 for p in workspace.rglob("*") if p.is_file()),
        "forbidden_artifact_count": 0,
    }


def unique_sha(root: Path, name: str, sha: str) -> Path:
    matches = [p.resolve() for p in root.rglob(name) if p.is_file() and sha256_file(p) == sha]
    if len(matches) != 1:
        raise TrackBOpsError(f"expected one {name} with SHA {sha} under {root}; found {matches}")
    return matches[0]


def find_v1(root: Path) -> tuple[Path, Path, Path]:
    manifest = unique_sha(root, "final_manifest.csv", DATASET_MANIFEST_SHA256)
    class_map = unique_sha(root, "class_to_idx.json", CLASS_MAP_SHA256)
    roots = []
    for parent in (manifest.parent.parent, manifest.parent, manifest.parent.parent.parent):
        image_root = parent / "dataset"
        if (image_root / "train").is_dir() and (image_root / "val").is_dir():
            roots.append(image_root.resolve())
    roots = list(dict.fromkeys(roots))
    if len(roots) != 1:
        raise TrackBOpsError(f"could not resolve one frozen Final-V1 image root: {roots}")
    return manifest, class_map, roots[0]


def resolve_bundle_file(bundle_root: Path, manifest: dict, key: str) -> Path:
    row = (manifest.get("files") or {}).get(key)
    if not isinstance(row, dict):
        raise TrackBOpsError(f"core bundle file key missing: {key}")
    path = (bundle_root / str(row.get("path", ""))).resolve()
    if bundle_root not in path.parents and path != bundle_root:
        raise TrackBOpsError(f"core bundle path escapes root: {key}")
    if not path.is_file():
        raise TrackBOpsError(f"core bundle file missing: {key}: {path}")
    expected = str(row.get("sha256", ""))
    if len(expected) != 64 or sha256_file(path) != expected:
        raise TrackBOpsError(f"core bundle file hash mismatch: {key}")
    return path


def validate_historical_cache(root: Path, core_root: Path) -> bool:
    manifests = list(root.rglob("TRACKB_INPUT_MANIFEST.json"))
    if len(manifests) != 1:
        return False
    try:
        hist = load_json(manifests[0])
        core = load_json(core_root / "TRACKB_INPUT_MANIFEST.json")
        expected_attestation = (core.get("files") or {}).get("code_attestation", {}).get("sha256")
        return (
            hist.get("role") == "historical_compare"
            and hist.get("coverage_scope") == "V1_TRAIN_VAL_ONLY"
            and int(hist.get("image_count", -1)) == 92744
            and hist.get("maximum_evidence_grade") == "EXT-S"
            and hist.get("v1_test_image_bytes_accessed") is False
            and hist.get("code_attestation_sha256") == expected_attestation
        )
    except Exception:
        return False


def normalize_downloaded_package(download_root: Path, target_root: Path) -> None:
    manifests = list(download_root.rglob("TRACKB_INPUT_MANIFEST.json"))
    if len(manifests) != 1:
        raise TrackBOpsError(f"cached package must contain exactly one TRACKB_INPUT_MANIFEST.json: {manifests}")
    source_root = manifests[0].parent
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)


def build_core(
    *,
    repo_root: Path,
    source_roots: dict[str, Path],
    core_root: Path,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "journal_extension/scripts/build_trackb_core_package.py"),
        "--repo-root", str(repo_root),
        "--final-v1-root", str(source_roots["final_v1"]),
        "--r07-s1-root", str(source_roots["r07_s1"]),
        "--r07-s2-root", str(source_roots["r07_s2"]),
        "--r07-s3-root", str(source_roots["r07_s3"]),
        "--dino-bundle-root", str(source_roots["dino_bundle"]),
        "--output-root", str(core_root),
    ]
    run_checked(cmd, cwd=repo_root, timeout=7200)


def build_historical(
    *,
    repo_root: Path,
    final_v1_root: Path,
    core_root: Path,
    historical_root: Path,
    device: str,
) -> None:
    v1_manifest, _class_map, image_root = find_v1(final_v1_root)
    core = load_json(core_root / "TRACKB_INPUT_MANIFEST.json")
    dino = resolve_bundle_file(core_root, core, "dino_checkpoint")
    factory = resolve_bundle_file(core_root, core, "dino_factory_manifest")
    execution_lock = resolve_bundle_file(core_root, core, "execution_lock")
    code_attestation = resolve_bundle_file(core_root, core, "code_attestation")
    factory_source_root = (core_root / str(core["dino_factory_source_root"])).resolve()
    cmd = [
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
    ]
    run_checked(cmd, cwd=repo_root, timeout=21600)


def prepare_candidate(repo_root: Path, role: str, package_root: Path) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "journal_extension/scripts/prepare_trackb_candidate_input.py"),
        "--role", role,
        "--package-root", str(package_root),
        "--data-root", "data",
        "--source-metadata-record", "SOURCE_METADATA.json",
    ]
    run_checked(cmd, cwd=repo_root, timeout=3600)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fully automated CropCop Track-B R07 Kaggle controller.")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--workspace", default="/kaggle/working/trackb_master")
    ap.add_argument("--scratch-root", default="/kaggle/tmp/cropcop_trackb_r07")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kaggle-owner", default=KAGGLE_OWNER_DEFAULT)
    ap.add_argument("--force-rebuild-historical", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    workspace = Path(args.workspace).resolve()
    scratch_root = Path(args.scratch_root).resolve()
    requested_kaggle_owner = str(args.kaggle_owner).strip()
    if workspace.exists() and any(workspace.iterdir()):
        raise TrackBOpsError(f"master workspace must be empty for a clean run: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=False)
    inputs_root = scratch_root / "inputs"
    sources_root = scratch_root / "sources"
    output_root = workspace / "trackb_r07"
    inputs_root.mkdir()
    sources_root.mkdir()

    stage("0 :: secrets and platform")
    secret_presence = configure_runtime_secrets()
    source_git_sha = run_checked(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        timeout=120,
    ).stdout.strip()
    if not source_git_sha:
        raise TrackBOpsError("could not bind repository HEAD")
    github_permission = verify_github_repository_push_access(
        repo_root,
        source_git_sha=source_git_sha,
    )
    kaggle_cli = ensure_kaggle_cli()
    kaggle_owner = verify_authenticated_kaggle_owner(requested_kaggle_owner)
    source_access = verify_kaggle_source_access(SOURCE_DATASETS)
    external_source_probe = probe_external_sources()
    historical_dataset = historical_dataset_slug(kaggle_owner)
    evidence_dataset = evidence_dataset_slug(kaggle_owner)
    receipt = {
        "schema_version": "2.0",
        "controller": "TRACKB_R07_MASTER_v2",
        "started_at_utc": utc_now(),
        "repository_head": source_git_sha,
        "secret_presence": secret_presence,
        "kaggle_cli": kaggle_cli,
        "github_permission_preflight": github_permission,
        "external_source_preflight": external_source_probe,
        "kaggle_source_access_preflight": source_access,
        "initial_disk_gb": {
            "persistent_working": disk_gb(workspace),
            "ephemeral_scratch": disk_gb(scratch_root),
        },
        "storage_policy": {
            "persistent_root": str(workspace),
            "scratch_root": str(scratch_root),
            "raw_external_images_persistent": False,
            "model_source_files_persistent": False,
        },
        "kaggle_owner": kaggle_owner,
        "protected_external_predictions_before_controller": False,
    }
    print(json.dumps({k: v for k, v in receipt.items() if k != "secret_presence"} | {"secret_presence": secret_presence}, indent=2))

    stage("1 :: automatic owned/private source acquisition")
    source_roots: dict[str, Path] = {}
    for key, slug in SOURCE_DATASETS.items():
        print(f"Downloading {key}: {slug}", flush=True)
        source_roots[key] = download_kaggle_dataset(slug, sources_root / key)
        print("scratch disk:", disk_gb(scratch_root), flush=True)

    stage("2 :: immutable core assembly")
    core_root = inputs_root / "core"
    build_core(repo_root=repo_root, source_roots=source_roots, core_root=core_root)
    core_manifest = load_json(core_root / "TRACKB_INPUT_MANIFEST.json")
    if core_manifest.get("role") != "core":
        raise TrackBOpsError("core builder did not produce role=core")

    stage("2.5 :: release redundant model-source downloads")
    for key in ("r07_s1", "r07_s2", "r07_s3", "dino_bundle"):
        shutil.rmtree(source_roots[key], ignore_errors=True)
    print("scratch disk after model-source cleanup:", disk_gb(scratch_root), flush=True)

    stage("3 :: historical comparison cache")
    historical_root = inputs_root / "historical_compare"
    used_cache = False
    if not args.force_rebuild_historical and kaggle_dataset_exists(historical_dataset):
        cache_download = scratch_root / "hist_cache_download"
        download_kaggle_dataset(historical_dataset, cache_download)
        if validate_historical_cache(cache_download, core_root):
            normalize_downloaded_package(cache_download, historical_root)
            used_cache = True
            print("Reused exact private historical comparison cache.", flush=True)
        shutil.rmtree(cache_download, ignore_errors=True)
    if not used_cache:
        build_historical(
            repo_root=repo_root,
            final_v1_root=source_roots["final_v1"],
            core_root=core_root,
            historical_root=historical_root,
            device=args.device,
        )
        if not validate_historical_cache(historical_root, core_root):
            raise TrackBOpsError("fresh historical comparison package failed cache contract")
        publish_private_kaggle_dataset(
            folder=historical_root,
            slug=historical_dataset,
            title="CropCop Track B R07 Historical Comparison Cache v2",
            version_message=f"safe V1 train+val comparison for {source_git_sha[:12]}",
            license_name="other",
        )
    receipt["historical_cache_reused"] = used_cache
    receipt["historical_cache_slug"] = historical_dataset

    shutil.rmtree(source_roots["final_v1"], ignore_errors=True)

    stage("4 :: garbage-collect source-model downloads")
    # The immutable core now contains exactly the validation surface/checkpoints/audit encoder needed by B0.
    # The historical package contains the only allowed post-closure historical representation.
    shutil.rmtree(sources_root, ignore_errors=True)
    print("scratch disk after source cleanup:", disk_gb(scratch_root), flush=True)

    stage("5 :: automatic external source acquisition and packaging")
    gvlid_root = inputs_root / "gvlid_v5"
    potato_root = inputs_root / "irish_potato"
    lineage_review = repo_root / "journal_extension/track_b_r07/TRACKB_EXTERNAL_LINEAGE_REVIEW_v1.json"
    if not lineage_review.is_file():
        raise TrackBOpsError(f"frozen external-lineage review missing: {lineage_review}")
    acquire_gvlid_v5(gvlid_root, lineage_review_path=lineage_review)
    prepare_candidate(repo_root, "gvlid_v5", gvlid_root)
    print("GVLiD packaged; scratch disk:", disk_gb(scratch_root), flush=True)
    acquire_irish_potato(potato_root, lineage_review_path=lineage_review)
    prepare_candidate(repo_root, "irish_potato", potato_root)
    print("Irish Potato packaged; scratch disk:", disk_gb(scratch_root), flush=True)

    stage("6 :: sealed Track-B execution")
    runner = repo_root / "journal_extension/scripts/run_trackb_r07.py"
    run_checked(
        [
            sys.executable, str(runner),
            "--input-root", str(inputs_root),
            "--output-root", str(output_root),
            "--device", args.device,
            "--workers", "4",
            "--mode", "all",
        ],
        cwd=repo_root,
        timeout=36000,
    )
    closure = load_json(output_root / "TRACKB_FINAL_CLOSURE.json")
    qa = load_json(output_root / "TRACKB_FINAL_QA.json")
    if closure.get("status") != "TRACK_B_CLOSED" or qa.get("status") != "PASS":
        raise TrackBOpsError("runner returned without terminal Track-B QA/closure")

    stage("7 :: restricted evidence archive to private Kaggle")
    restricted = prepare_private_evidence_folder(output_root, scratch_root / "restricted_archive")
    private_receipt = publish_private_kaggle_dataset(
        folder=restricted,
        slug=evidence_dataset,
        title="CropCop Track B R07 Restricted Evidence",
        version_message=f"Track-B closure {closure['closure_sha256'][:16]}",
        license_name="other",
    )
    receipt["private_kaggle_evidence"] = private_receipt
    receipt["closure_sha256"] = closure["closure_sha256"]
    receipt["final_qa_sha256"] = qa["qa_sha256"]
    receipt["scientific_closure_durable_before_github_publication"] = True

    # Persist a checkpoint receipt before the non-scientific publication layer.
    receipt["status"] = "TRACK_B_CLOSED_PRIVATE_EVIDENCE_ARCHIVED"
    receipt["github_public_evidence"] = None
    (output_root / "TRACKB_AUTOMATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )

    stage("8 :: safe GitHub evidence publication")
    github_receipt = None
    github_errors = []
    for attempt, delay in enumerate((0, 5, 15, 30), start=1):
        if delay:
            time.sleep(delay)
        try:
            github_receipt = publish_public_trackb_evidence(
                repo_root=repo_root,
                source_git_sha=source_git_sha,
                output_root=output_root,
            )
            break
        except Exception as exc:
            message = redact(str(exc))
            github_errors.append(
                {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "message": message[-1800:],
                }
            )
            print(
                f"GitHub evidence publication attempt {attempt}/4 failed: "
                f"{type(exc).__name__}: {message[-600:]}",
                flush=True,
            )

    if github_receipt is None:
        receipt["github_publication_errors"] = github_errors
        receipt["completed_at_utc"] = utc_now()
        receipt["final_disk_gb"] = disk_gb(workspace)
        receipt["status"] = "TRACK_B_CLOSED_PRIVATE_EVIDENCE_ARCHIVED_GITHUB_PUBLICATION_FAILED"
        receipt["manual_publication_steps_required"] = 1
        (output_root / "TRACKB_AUTOMATION_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        raise TrackBOpsError(
            "Track B scientific closure is complete and restricted evidence is safely "
            "archived on private Kaggle, but GitHub public-safe evidence publication "
            "failed after 4 attempts. Do not rerun science; repair GitHub connectivity/"
            "credential and publish from the preserved closure evidence."
        )

    receipt["github_public_evidence"] = github_receipt
    print(json.dumps(github_receipt, indent=2), flush=True)

    receipt["completed_at_utc"] = utc_now()
    shutil.rmtree(scratch_root, ignore_errors=True)
    receipt["persistent_hygiene"] = assert_persistent_output_hygiene(workspace, output_root)
    receipt["final_disk_gb"] = disk_gb(workspace)
    receipt["status"] = "PASS_AUTOMATED_TRACK_B_COMPLETE"
    receipt["manual_publication_steps_required"] = 0
    receipt["raw_external_images_committed_to_github"] = False
    receipt["model_weights_committed_to_github"] = False
    (output_root / "TRACKB_AUTOMATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
