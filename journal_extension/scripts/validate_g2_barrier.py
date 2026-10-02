from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.g2 import REQUIRED_CALIBRATIONS, write_g2_barrier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.summaries_dir)
    summaries = []
    for cid in REQUIRED_CALIBRATIONS:
        path = root / f"{cid}.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    barrier = write_g2_barrier(args.output, summaries)
    print(json.dumps(barrier, indent=2, sort_keys=True))
    return 0 if barrier["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
