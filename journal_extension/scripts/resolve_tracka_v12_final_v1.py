from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.tracka_v12_final_v1 import resolve_final_v1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--certificate", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != args.analysis_source_git_commit:
        raise SystemExit(
            f"Final-V1 resolver exact-head mismatch: expected={args.analysis_source_git_commit}, observed={observed}"
        )

    resolution, diagnostics = resolve_final_v1(output_root=args.output_root)
    certificate = resolution.certificate()
    certificate.update(
        {
            "analysis_source_git_commit": observed,
            "discovery_diagnostics": diagnostics,
        }
    )
    atomic_write_json(args.certificate, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
