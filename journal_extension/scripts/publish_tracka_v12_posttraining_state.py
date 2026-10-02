from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.publication import audit_public_files, publish_to_github_branch

PUBLIC_BASENAMES = {
    "DIRECT_STATE_EVIDENCE_GATE.json",
    "robustness_and_replay.json",
    "efficiency.json",
    "XAI_EVIDENCE_GATE.json",
    "AUXILIARY_STATE_EVIDENCE_GATE.json",
    "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json",
    "POSTTRAINING_STATE_COMPLETION.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--state-root", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    state_root = Path(args.state_root).resolve()
    if not state_root.is_dir():
        raise SystemExit("state evidence root missing")
    if len(args.analysis_source_git_commit) != 40:
        raise SystemExit("analysis source Git SHA must be full length")

    candidates = sorted(
        path
        for path in state_root.rglob("*.json")
        if path.name in PUBLIC_BASENAMES
    )
    names = [path.name for path in candidates]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate public evidence basenames are ambiguous: {names}")
    if "POSTTRAINING_STATE_COMPLETION.json" not in names:
        raise SystemExit("state completion manifest must exist before public publication")
    approved = audit_public_files(candidates)

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        staged = []
        hashes = {}
        for src in approved:
            dst = staging / src.name
            dst.write_bytes(src.read_bytes())
            staged.append(dst)
            hashes[src.name] = sha256_file(src)
        manifest = {
            "schema_version": "1.0",
            "status": "PASS",
            "publication_kind": "track_a_posttraining_state",
            "experiment_id": args.experiment_id,
            "run_id": args.run_id,
            "analysis_source_git_commit": args.analysis_source_git_commit,
            "public_file_sha256": hashes,
            "public_file_count": len(hashes),
            "private_material_published": False,
        }
        manifest["publication_manifest_sha256"] = sha256_json(manifest)
        manifest_path = staging / "POSTTRAINING_PUBLICATION_MANIFEST.json"
        atomic_write_json(manifest_path, manifest)
        audit_public_files([manifest_path])
        manifest_file_sha256 = sha256_file(manifest_path)
        staged.append(manifest_path)

        branch = publish_to_github_branch(
            repo_dir=repo,
            source_git_sha=args.analysis_source_git_commit,
            run_id=args.run_id,
            files=staged,
            destination_prefix="journal_extension/evidence/public/track_a/posttraining",
        )

    result = {
        **manifest,
        "status": "PASS",
        "publication_branch": branch,
        "publication_manifest_file_sha256": manifest_file_sha256,
    }
    result["publication_certificate_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
