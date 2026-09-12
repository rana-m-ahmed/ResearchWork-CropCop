from __future__ import annotations
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--source", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--evidence", required=True)
ap.add_argument("--predictions", required=True)
args = ap.parse_args()

src = Path(args.repo) / "journal_extension" / "src"
sys.path.insert(0, str(src))
from cropcop_je.publication import publish_to_github_branch

print(publish_to_github_branch(
    repo_dir=args.repo,
    source_git_sha=args.source,
    run_id=args.run_id,
    files=[args.evidence, args.predictions],
))
