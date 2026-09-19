from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1_publication import preflight_private_target
from cropcop_je.hashing import sha256_file, sha256_json

DENIED_NAMES = {"dataset-metadata.json", "POSTTRAINING_DURABILITY_MARKER.json"}


def file_manifest(root: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if Path(rel).name in DENIED_NAMES:
            continue
        rows[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not rows:
        raise RuntimeError("post-training evidence tree is empty")
    return rows


def run(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    cp = subprocess.run(args, env=env, check=False, capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()[-1500:]
        raise RuntimeError(f"command failed rc={cp.returncode}: {' '.join(args)} :: {detail}")
    return cp


def current_version(locator: str) -> int:
    state = preflight_private_target(locator)
    if state.get("authoritative_is_private") is not True:
        raise RuntimeError("post-training durability target is not authoritative private")
    value = state.get("current_version_number")
    if value is None:
        raise RuntimeError("post-training durability target lacks current version")
    return int(value)


def download(locator: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", locator, "-p", str(target), "--unzip", "-q"])


def verify_download(
    root: Path,
    *,
    run_id: str,
    experiment_id: str,
    nonce: str,
    manifest_sha: str,
    generation_kind: str,
) -> dict:
    marker_path = root / "POSTTRAINING_DURABILITY_MARKER.json"
    if not marker_path.is_file():
        raise RuntimeError("downloaded evidence generation lacks durability marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "sync_nonce": nonce,
        "evidence_manifest_sha256": manifest_sha,
        "generation_kind": generation_kind,
        "complete": True,
    }
    for field, value in expected.items():
        if marker.get(field) != value:
            raise RuntimeError(f"downloaded evidence generation marker mismatch: {field}")
    manifest = marker.get("files") or {}
    if sha256_json(manifest) != manifest_sha:
        raise RuntimeError("downloaded evidence marker manifest self-hash mismatch")
    for rel, row in manifest.items():
        path = root / rel
        if not path.is_file():
            raise RuntimeError(f"downloaded evidence file missing: {rel}")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"downloaded evidence file hash/size mismatch: {rel}")
    return marker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--dataset-locator", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generation-kind", choices=["partial", "final"], default="final")
    ap.add_argument("--settle-timeout-seconds", type=float, default=900)
    args = ap.parse_args()

    source = Path(args.source_dir).resolve()
    if not source.is_dir():
        raise SystemExit("post-training evidence source directory is missing")
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    owner, _, _dataset = args.dataset_locator.partition("/")
    if not username or owner.casefold() != username.casefold():
        raise SystemExit("post-training private evidence dataset owner must equal authenticated KAGGLE_USERNAME")
    if len(args.analysis_source_git_commit) != 40:
        raise SystemExit("analysis source Git SHA must be full length")

    before = current_version(args.dataset_locator)
    nonce = uuid.uuid4().hex
    source_manifest = file_manifest(source)
    manifest_sha = sha256_json(source_manifest)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        for path in source.rglob("*"):
            if path.is_file():
                rel = path.relative_to(source)
                dst = staging / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
        marker = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "experiment_id": args.experiment_id,
            "analysis_source_git_commit": args.analysis_source_git_commit,
            "generation_kind": args.generation_kind,
            "sync_nonce": nonce,
            "previous_version_number": before,
            "files": source_manifest,
            "evidence_manifest_sha256": manifest_sha,
            "complete": True,
        }
        atomic_write_json(staging / "POSTTRAINING_DURABILITY_MARKER.json", marker)
        atomic_write_json(
            staging / "dataset-metadata.json",
            {
                "title": args.dataset_locator.split("/", 1)[1],
                "id": args.dataset_locator,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
        )
        run([
            "kaggle", "datasets", "version", "-p", str(staging),
            "-m", f"Track-A post-training evidence {args.experiment_id}", "-q", "-r", "zip"
        ])

    deadline = time.monotonic() + args.settle_timeout_seconds
    after = before
    while time.monotonic() < deadline:
        after = current_version(args.dataset_locator)
        if after > before:
            break
        time.sleep(5)
    if after <= before:
        raise SystemExit("private evidence dataset did not expose a newer generation before timeout")

    roundtrip_marker = None
    roundtrip_sha = None
    last_error = None
    while time.monotonic() < deadline:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            try:
                download(args.dataset_locator, root)
                roundtrip_marker = verify_download(
                    root,
                    run_id=args.run_id,
                    experiment_id=args.experiment_id,
                    nonce=nonce,
                    manifest_sha=manifest_sha,
                    generation_kind=args.generation_kind,
                )
                roundtrip_sha = sha256_file(root / "POSTTRAINING_DURABILITY_MARKER.json")
                break
            except Exception as exc:
                last_error = exc
        time.sleep(5)
    if roundtrip_marker is None:
        raise SystemExit(f"private evidence generation never round-trip verified: {last_error}")

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "sync_kind": "track_a_posttraining_private_evidence",
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "analysis_source_git_commit": args.analysis_source_git_commit,
        "generation_kind": args.generation_kind,
        "dataset_locator": args.dataset_locator,
        "owner_matches_authenticated_user": True,
        "previous_version_number": before,
        "confirmed_version_number": after,
        "evidence_file_count": len(source_manifest),
        "evidence_manifest_sha256": manifest_sha,
        "roundtrip_marker_sha256": roundtrip_sha,
        "generation_roundtrip_verified": True,
        "private_dataset_verified": True,
    }
    result["sync_certificate_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
