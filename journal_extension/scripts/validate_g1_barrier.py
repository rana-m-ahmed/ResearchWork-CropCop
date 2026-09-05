from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.envelope import AMENDMENT_ID, AMENDMENT_SHA256
from cropcop_je.g1 import validate_mounted_g1
from cropcop_je.hashing import sha256_json
from cropcop_je.smoke_handoff import validate_terminal_dual_gpu_smoke_evidence
from cropcop_je.source_state import verify_clean_source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--g1-bundle-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--teacher-factory-root", default="")
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--infra-smoke-evidence", required=True)
    ap.add_argument("--dual-gpu-smoke-evidence", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.g1_bundle_dir).resolve()
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.exists():
        raise SystemExit("G1_MODEL_IDENTITY_SEAL.json is missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))

    errors = []
    try:
        source = verify_clean_source(repo, authorized_source_sha=args.authorized_source_sha, output_roots=[bundle])
    except Exception as exc:
        source = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))

    errors.extend(validate_mounted_g1(
        seal,
        authorized_source_sha=args.authorized_source_sha,
        manifest_path=args.manifest,
        class_map_path=args.class_map,
        pretrained_path=args.pretrained,
        pretrained_provenance_path=bundle / "evidence/MNV4_PRETRAINED_PROVENANCE.json",
        pair_dir=bundle / "private",
        evidence_dir=bundle / "evidence",
        teacher_checkpoint_path=args.teacher_checkpoint,
        teacher_factory_manifest_path=bundle / "evidence/TEACHER_FACTORY_BUNDLE.json",
        teacher_factory_source_root=(args.teacher_factory_root or repo),
        teacher_class_order_evidence_path=bundle / "evidence/TEACHER_CLASS_ORDER_EVIDENCE.json",
        dependency_lock_path=repo / args.dependency_lock,
        infra_smoke_evidence_path=args.infra_smoke_evidence,
        repo_root=repo,
    ))
    dependency = json.loads((repo / args.dependency_lock).read_text(encoding="utf-8"))
    smoke = json.loads(Path(args.infra_smoke_evidence).read_text(encoding="utf-8"))
    dual_smoke = json.loads(Path(args.dual_gpu_smoke_evidence).read_text(encoding="utf-8"))
    errors.extend(
        "dual-GPU-smoke: " + message
        for message in validate_terminal_dual_gpu_smoke_evidence(
            dual_smoke,
            expected_source_sha=args.authorized_source_sha,
            expected_dependency_lock_sha256=dependency.get("dependency_lock_sha256", ""),
            expected_amendment_id=AMENDMENT_ID,
            expected_amendment_sha256=AMENDMENT_SHA256,
            expected_smoke_b_evidence_sha256=sha256_json(smoke),
            require_batch=True,
        )
    )
    report = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "g1_seal_sha256": seal.get("g1_seal_sha256"),
        "source_git_sha": seal.get("source_git_sha"),
        "dependency_lock_sha256": seal.get("dependency_lock_sha256"),
        "source_state": source,
        "errors": errors,
    }
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
