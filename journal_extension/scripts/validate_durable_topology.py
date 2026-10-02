from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from cropcop_je.persistence import validate_durable_locator_template


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["filesystem", "kaggle-dataset"], required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--run-id", action="append", dest="run_ids", required=True)
    args = ap.parse_args()
    resolved = validate_durable_locator_template(args.kind, args.template, args.run_ids)
    print(json.dumps({"status": "PASS", "kind": args.kind, "resolved": resolved}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
