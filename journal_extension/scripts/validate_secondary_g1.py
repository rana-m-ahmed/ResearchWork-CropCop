from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from cropcop_je.secondary import validate_secondary_g1_bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    args = ap.parse_args()
    seal, errors = validate_secondary_g1_bundle(args.bundle_dir)
    report = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "secondary_g1_seal_sha256": seal.get("secondary_g1_seal_sha256"),
        "source_git_sha": seal.get("source_git_sha"),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
