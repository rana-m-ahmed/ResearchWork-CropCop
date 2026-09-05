from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import validate_g1_seal_object
from cropcop_je.source_state import verify_clean_source
from cropcop_je.validate import validate_static


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--g1-seal", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--json-report", default="")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    errors = []
    static = validate_static(repo)
    errors.extend(static.get("errors", []))
    try:
        source = verify_clean_source(
            repo,
            authorized_source_sha=args.authorized_source_sha,
            output_roots=[args.output_root],
        )
    except Exception as exc:
        source = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))
    seal = json.loads(Path(args.g1_seal).read_text(encoding="utf-8"))
    errors.extend(validate_g1_seal_object(seal))
    if seal.get("source_git_sha") != args.authorized_source_sha:
        errors.append("G1 seal source differs from authorized prelaunch source")
    report = {
        "schema_version": "2.0",
        "status": "PASS" if not errors else "FAIL",
        "static": static,
        "source": source,
        "g1_seal_sha256": seal.get("g1_seal_sha256"),
        "errors": errors,
    }
    if args.json_report:
        atomic_write_json(args.json_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
