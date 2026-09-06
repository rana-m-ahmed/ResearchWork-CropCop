from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.g1_barrier import G1BarrierInputs, validate_and_write


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--g1-bundle-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--infra-smoke-evidence", required=True)
    ap.add_argument("--dual-gpu-smoke-evidence", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    dep = Path(args.dependency_lock)
    if not dep.is_absolute():
        dep = repo / dep
    inputs = G1BarrierInputs(
        repo_root=repo,
        authorized_source_sha=args.authorized_source_sha,
        bundle_dir=Path(args.g1_bundle_dir).resolve(),
        manifest=Path(args.manifest).resolve(),
        class_map=Path(args.class_map).resolve(),
        infra_smoke_evidence=Path(args.infra_smoke_evidence).resolve(),
        dual_gpu_smoke_evidence=Path(args.dual_gpu_smoke_evidence).resolve(),
        dependency_lock=dep.resolve(),
    )
    report = validate_and_write(inputs, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
