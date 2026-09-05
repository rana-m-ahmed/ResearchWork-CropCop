from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.runlog import validate_run_record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    src = Path(args.run_record)
    record = json.loads(src.read_text(encoding="utf-8"))
    validate_run_record(record)
    if record["status"] not in {"PASS", "FAIL", "INCONCLUSIVE", "INTERRUPTED"}:
        raise SystemExit("only terminal run records may be ingested")
    dst = Path(args.repo_root) / "journal_extension/runs" / f"{record['run_id']}.json"
    if dst.exists():
        old = json.loads(dst.read_text(encoding="utf-8"))
        if old != record:
            raise SystemExit(f"run ID collision with different evidence: {record['run_id']}")
        print(dst)
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
