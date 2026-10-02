from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from cropcop_je.publication import publish_to_github_branch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--source-git-sha", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--file", action="append", dest="files", required=True)
    args = ap.parse_args()
    branch = publish_to_github_branch(
        repo_dir=args.repo_dir,
        source_git_sha=args.source_git_sha,
        run_id=args.run_id,
        files=args.files,
    )
    print(json.dumps({"status": "PASS", "branch": branch}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
