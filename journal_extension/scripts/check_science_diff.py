from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.science_diff import validate_science_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--json-report", default="")
    args = ap.parse_args()
    report = validate_science_diff(Path(args.repo_root))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_report:
        Path(args.json_report).write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
