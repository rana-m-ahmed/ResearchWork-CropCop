from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1_publication import preflight_private_target
from cropcop_je.hashing import sha256_file, sha256_json


def run(args: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = env.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    cp = subprocess.run(args, env=env, check=False, capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()[-1500:]
        raise RuntimeError(f"command failed rc={cp.returncode}: {' '.join(args)} :: {detail}")
    return cp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-locator", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--destination-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    owner, _, _dataset = args.dataset_locator.partition("/")
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not username or owner.casefold() != username.casefold():
        raise SystemExit("post-training evidence restore requires locator owner == authenticated KAGGLE_USERNAME")
    if len(args.analysis_source_git_commit) != 40:
        raise SystemExit("analysis source Git SHA must be full length")
    state = preflight_private_target(args.dataset_locator)
    if state.get("authoritative_is_private") is not True:
        raise SystemExit("post-training evidence restore target is not authoritatively private")

    destination = Path(args.destination_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run(["kaggle", "datasets", "download", "-d", args.dataset_locator, "-p", str(root), "--unzip", "-q"])
        marker_path = root / "POSTTRAINING_DURABILITY_MARKER.json"
        if not marker_path.is_file():
            result = {
                "schema_version": "1.0",
                "status": "EMPTY",
                "restore_kind": "track_a_posttraining_private_evidence",
                "experiment_id": args.experiment_id,
                "run_id": args.run_id,
                "analysis_source_git_commit": args.analysis_source_git_commit,
                "dataset_locator": args.dataset_locator,
                "current_version_number": state.get("current_version_number"),
                "restored_file_count": 0,
                "generation_kind": None,
            }
            result["restore_certificate_sha256"] = sha256_json(result)
            atomic_write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "1.0",
            "run_id": args.run_id,
            "experiment_id": args.experiment_id,
            "analysis_source_git_commit": args.analysis_source_git_commit,
            "complete": True,
        }
        for field, value in expected.items():
            if marker.get(field) != value:
                raise SystemExit(f"post-training evidence restore marker mismatch: {field}")
        generation_kind = marker.get("generation_kind")
        if generation_kind not in {"partial", "final"}:
            raise SystemExit("post-training evidence restore marker has unsupported generation_kind")
        manifest = marker.get("files") or {}
        manifest_sha = marker.get("evidence_manifest_sha256")
        if manifest_sha != sha256_json(manifest):
            raise SystemExit("post-training evidence restore manifest self-hash mismatch")

        copied = 0
        for rel, row in sorted(manifest.items()):
            source = root / rel
            if not source.is_file():
                raise SystemExit(f"post-training evidence restore source missing: {rel}")
            if source.stat().st_size != int(row["bytes"]) or sha256_file(source) != row["sha256"]:
                raise SystemExit(f"post-training evidence restore source hash/size mismatch: {rel}")
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != source.stat().st_size or sha256_file(target) != row["sha256"]:
                    raise SystemExit(f"post-training evidence restore refuses conflicting local file: {rel}")
            else:
                shutil.copy2(source, target)
                copied += 1

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "restore_kind": "track_a_posttraining_private_evidence",
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "analysis_source_git_commit": args.analysis_source_git_commit,
        "dataset_locator": args.dataset_locator,
        "current_version_number": state.get("current_version_number"),
        "generation_kind": generation_kind,
        "evidence_manifest_sha256": manifest_sha,
        "restored_file_count": copied,
        "verified_manifest_file_count": len(manifest),
        "generation_verified": True,
    }
    result["restore_certificate_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
