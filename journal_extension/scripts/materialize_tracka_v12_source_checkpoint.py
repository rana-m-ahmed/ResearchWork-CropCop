from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_json
from cropcop_je.tracka_v12_source_materialization import (
    materialize_source_checkpoint,
    restore_for_historical_availability,
)


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--mode", choices=["historical-availability", "full"], required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != args.analysis_source_git_commit:
        raise SystemExit(
            f"source materializer exact-checkout mismatch: "
            f"expected={args.analysis_source_git_commit}, observed={observed}"
        )

    record = load_json(args.run_record)
    if args.mode == "historical-availability":
        result = restore_for_historical_availability(
            record=record,
            checkpoint_root=args.checkpoint_root,
        )
    else:
        result = materialize_source_checkpoint(
            record=record,
            checkpoint_root=args.checkpoint_root,
        )

    result = {
        **result,
        "analysis_source_git_commit": observed,
        "source_run_record": str(Path(args.run_record).resolve()),
        "source_run_record_sha256": __import__(
            "cropcop_je.hashing", fromlist=["sha256_file"]
        ).sha256_file(args.run_record),
        "mode": args.mode,
    }
    result["certificate_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
