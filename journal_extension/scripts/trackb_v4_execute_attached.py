from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.trackb_r07 import TrackBError, load_json
from cropcop_je.trackb_r07_ops import (
    TrackBOpsError,
    configure_runtime_secrets,
    evidence_dataset_slug,
    prepare_private_evidence_folder,
    publish_private_kaggle_dataset,
    run_checked,
    verify_authenticated_kaggle_owner,
)


def stage(name: str) -> None:
    print(f"\n{'=' * 18} {name} {'=' * 18}", flush=True)


def _find_single(root: Path, name: str) -> Path:
    matches = sorted(path.resolve() for path in root.glob(f"**/{name}") if path.is_file())
    if len(matches) != 1:
        raise TrackBOpsError(f"expected exactly one {name} under {root}; found {matches}")
    return matches[0]


def _read_pair_receipts(input_root: Path) -> tuple[dict, dict]:
    infra_path = _find_single(input_root, "TRACKB_INFRASTRUCTURE_BUNDLE.json")
    external_path = _find_single(input_root, "TRACKB_EXTERNAL_BUNDLE.json")
    infra = load_json(infra_path)
    external = load_json(external_path)
    if infra.get("bundle_role") != "TRACKB_INFRASTRUCTURE":
        raise TrackBOpsError("infrastructure bundle receipt role mismatch")
    if external.get("bundle_role") != "TRACKB_EXTERNAL":
        raise TrackBOpsError("external bundle receipt role mismatch")
    for field in (
        "materialization_id",
        "repository_source_sha",
        "scientific_execution_lock_sha256",
        "scientific_code_attestation_sha256",
        "role_manifest_sha256",
        "role_content_identity",
    ):
        if infra.get(field) != external.get(field):
            raise TrackBOpsError(f"Track-B attached bundles are not paired: mismatch={field}")
    materialization_id = str(infra.get("materialization_id", ""))
    if len(materialization_id) != 64:
        raise TrackBOpsError("invalid Track-B materialization_id")
    return infra, external


def _role_content_identity(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TrackBOpsError(f"attached Track-B role contains a symlink: {path}")
        if not path.is_file():
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    if not rows:
        raise TrackBOpsError(f"attached Track-B role payload is empty: {root}")
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "content_sha256": sha256_json(rows),
    }


def _discover_role_manifests(input_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(input_root.glob("**/TRACKB_INPUT_MANIFEST.json")):
        obj = load_json(path)
        role = str(obj.get("role", "")).strip()
        if not role:
            raise TrackBOpsError(f"Track-B input manifest lacks role: {path}")
        if role in found:
            raise TrackBOpsError(f"duplicate Track-B attached role {role}: {found[role]} / {path}")
        found[role] = path.resolve()
    required = {"core", "historical_compare", "gvlid_v5", "irish_potato"}
    if set(found) != required:
        raise TrackBOpsError(
            f"Track-B attached role mismatch: expected={sorted(required)}, observed={sorted(found)}"
        )
    return found


def _validate_pairing(input_root: Path) -> dict:
    infra, external = _read_pair_receipts(input_root)
    roles = _discover_role_manifests(input_root)
    declared = infra.get("role_manifest_sha256") or {}
    observed = {role: sha256_file(path) for role, path in roles.items()}
    if declared != observed:
        raise TrackBOpsError(
            "Track-B role-manifest hashes do not match the paired bundle receipts: "
            f"declared={declared}, observed={observed}"
        )

    declared_content = infra.get("role_content_identity") or {}
    observed_content = {
        role: _role_content_identity(path.parent)
        for role, path in roles.items()
    }
    if declared_content != observed_content:
        raise TrackBOpsError(
            "Track-B attached role payload bytes differ from the paired materialization: "
            f"declared={declared_content}, observed={observed_content}"
        )

    core_manifest = load_json(roles["core"])
    core_root = roles["core"].parent
    files = core_manifest.get("files") or {}
    for key, declared_hash_field in (
        ("execution_lock", "scientific_execution_lock_sha256"),
        ("code_attestation", "scientific_code_attestation_sha256"),
    ):
        row = files.get(key)
        if not isinstance(row, dict):
            raise TrackBOpsError(f"core manifest lacks required {key}")
        path = (core_root / str(row.get("path", ""))).resolve()
        if not path.is_file():
            raise TrackBOpsError(f"core {key} missing: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != infra.get(declared_hash_field):
            raise TrackBOpsError(f"paired bundle {key} identity mismatch")

    repo_rel = str(core_manifest.get("repository_root", "")).strip()
    repo_root = (core_root / repo_rel).resolve()
    if not (repo_root / "journal_extension").is_dir():
        raise TrackBOpsError(f"embedded Track-B repository snapshot missing: {repo_root}")

    return {
        "materialization_id": infra["materialization_id"],
        "repository_source_sha": infra["repository_source_sha"],
        "role_manifest_sha256": observed,
        "role_content_identity": observed_content,
        "repo_root": repo_root,
        "core_manifest_path": roles["core"],
    }


def _run_qualification(
    *,
    repo_root: Path,
    input_root: Path,
    output_root: Path,
    scratch_root: Path,
    source_git_sha: str,
    device: str,
) -> dict:
    runner = repo_root / "journal_extension/scripts/run_trackb_r07.py"
    validator = repo_root / "journal_extension/scripts/validate_trackb_preinference_qualification.py"
    run_checked(
        [
            sys.executable, str(runner),
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--scratch-root", str(scratch_root / "audit"),
            "--device", device,
            "--workers", "4",
            "--mode", "qualification",
            "--source-git-sha", source_git_sha,
        ],
        cwd=repo_root,
        timeout=36000,
    )
    run_checked(
        [
            sys.executable, str(validator),
            "--input-root", str(input_root),
            "--output-root", str(output_root),
        ],
        cwd=repo_root,
        timeout=3600,
    )
    qualification = load_json(output_root / "TRACKB_PREINFERENCE_QUALIFICATION.json")
    qa = load_json(output_root / "TRACKB_PREINFERENCE_QA.json")
    if qualification.get("status") != "PASS_PREDICTION_BLIND_QUALIFICATION":
        raise TrackBOpsError("prediction-blind qualification did not PASS")
    if qa.get("status") != "PASS_INDEPENDENT_PREINFERENCE_QA":
        raise TrackBOpsError("independent pre-inference QA did not PASS")
    if int(qa.get("protected_external_prediction_count", -1)) != 0:
        raise TrackBOpsError("qualification produced protected external predictions")
    if qa.get("v1_test_accessed") is not False:
        raise TrackBOpsError("qualification reports consumed V1-test access")
    if qualification.get("qualification_science_sha256") != qa.get("qualification_science_sha256"):
        raise TrackBOpsError("qualification science digest differs from independent QA digest")
    return {
        "status": "PASS_TRACKB_PREINFERENCE_QUALIFICATION",
        "qualification_sha256": qualification["qualification_sha256"],
        "qualification_science_sha256": qualification["qualification_science_sha256"],
        "preinference_qa_sha256": qa["qa_sha256"],
        "protected_external_prediction_count": 0,
        "v1_test_accessed": False,
    }


def _run_claim(
    *,
    repo_root: Path,
    input_root: Path,
    output_root: Path,
    scratch_root: Path,
    source_git_sha: str,
    device: str,
    authorized_sha: str,
    kaggle_owner: str,
) -> dict:
    if len(authorized_sha) != 64 or any(ch not in "0123456789abcdef" for ch in authorized_sha):
        raise TrackBOpsError("claim mode requires a 64-character reviewed qualification science SHA-256")

    configure_runtime_secrets(require_github=False)
    owner = verify_authenticated_kaggle_owner(kaggle_owner)
    attempt_slug = f"{owner}/cropcop-trackb-r07-attempt-ledger-v4"
    attempt_id = f"TB4-{source_git_sha[:12]}-{int(time.time())}"
    runner = repo_root / "journal_extension/scripts/run_trackb_r07.py"

    run_checked(
        [
            sys.executable, str(runner),
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--scratch-root", str(scratch_root / "audit"),
            "--device", device,
            "--workers", "4",
            "--mode", "all",
            "--source-git-sha", source_git_sha,
            "--attempt-dataset-slug", attempt_slug,
            "--attempt-id", attempt_id,
            "--authorized-qualification-science-sha256", authorized_sha,
        ],
        cwd=repo_root,
        timeout=36000,
    )

    closure = load_json(output_root / "TRACKB_FINAL_CLOSURE.json")
    qa = load_json(output_root / "TRACKB_FINAL_QA.json")
    if closure.get("status") != "TRACK_B_CLOSED" or qa.get("status") != "PASS":
        raise TrackBOpsError("protected Track-B run did not reach terminal closure + QA PASS")
    if closure.get("v1_test_accessed") is not False:
        raise TrackBOpsError("protected Track-B closure reports V1-test access")

    restricted = prepare_private_evidence_folder(output_root, scratch_root / "restricted_archive")
    evidence_slug = evidence_dataset_slug(owner) + "-v4"
    private_receipt = publish_private_kaggle_dataset(
        folder=restricted,
        slug=evidence_slug,
        title="CropCop Track B R07 Restricted Evidence v4",
        version_message=f"Track-B closure {str(closure['closure_sha256'])[:16]}",
        license_name="other",
        full_roundtrip=True,
    )

    return {
        "status": "PASS_TRACKB_CLOSED_PRIVATE_ARCHIVED",
        "owner": owner,
        "attempt_id": attempt_id,
        "attempt_dataset_slug": attempt_slug,
        "closure_sha256": closure["closure_sha256"],
        "trackb_science_sha256": closure["trackb_science_sha256"],
        "final_qa_sha256": qa["qa_sha256"],
        "private_evidence": private_receipt,
        "public_evidence_zip": str(output_root / "TRACKB_PUBLIC_EVIDENCE.zip"),
        "v1_test_accessed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CropCop Track-B v4 attached-input execution controller."
    )
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--output-root", default="/kaggle/working/trackb_r07")
    ap.add_argument("--scratch-root", default="/kaggle/tmp/cropcop_trackb_r07_v4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=["qualification", "claim"], default="qualification")
    ap.add_argument("--authorized-qualification-science-sha256", default="")
    ap.add_argument("--kaggle-owner", default="AUTO")
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    scratch_root = Path(args.scratch_root).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=False)

    stage("0 :: paired attached-input validation")
    paired = _validate_pairing(input_root)
    repo_root = Path(paired["repo_root"]).resolve()
    source_git_sha = str(paired["repository_source_sha"])
    if len(source_git_sha) != 40:
        raise TrackBOpsError("paired input bundles lack an exact repository source SHA")

    stage("1 :: Track-B scientific execution")
    if args.mode == "qualification":
        result = _run_qualification(
            repo_root=repo_root,
            input_root=input_root,
            output_root=output_root,
            scratch_root=scratch_root,
            source_git_sha=source_git_sha,
            device=args.device,
        )
    else:
        result = _run_claim(
            repo_root=repo_root,
            input_root=input_root,
            output_root=output_root,
            scratch_root=scratch_root,
            source_git_sha=source_git_sha,
            device=args.device,
            authorized_sha=str(args.authorized_qualification_science_sha256).strip().lower(),
            kaggle_owner=args.kaggle_owner,
        )

    receipt = {
        "schema_version": "1.0",
        "controller": "TRACKB_V4_ATTACHED_EXECUTION",
        "mode": args.mode,
        "materialization_id": paired["materialization_id"],
        "repository_source_sha": source_git_sha,
        "role_manifest_sha256": paired["role_manifest_sha256"],
        "role_content_identity": paired["role_content_identity"],
        **result,
    }
    (output_root / "TRACKB_V4_EXECUTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
