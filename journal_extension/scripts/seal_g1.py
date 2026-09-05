from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import (
    AUTHORITY_ID, AUTHORITY_SHA256, CLASS_MAP_SHA256, MANIFEST_SHA256,
    MNV4_MODEL_NAME, PAIR_SPECS, TEACHER_SHA256, TIMM_VERSION,
    assert_g1_creation_target_fresh, g1_seal_hash, validate_dependency_environment, validate_dependency_lock_object,
    validate_pretrained_provenance, validate_teacher_class_order_evidence,
    validate_teacher_factory_bundle,
)
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.models import create_student_from_pretrained, save_pair_initialization
from cropcop_je.source_state import verify_clean_source


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--pretrained-provenance", required=True)
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--teacher-factory", required=True)
    ap.add_argument("--teacher-factory-manifest", required=True)
    ap.add_argument("--teacher-factory-root", default="")
    ap.add_argument("--teacher-class-order-evidence", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--infra-smoke-evidence", required=True)
    ap.add_argument("--bundle-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.bundle_dir).resolve()
    verify_clean_source(repo, authorized_source_sha=args.authorized_source_sha, output_roots=[bundle])

    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    private_dir = bundle / "private"
    evidence_dir = bundle / "evidence"
    assert_g1_creation_target_fresh(bundle)

    smoke = _load(args.infra_smoke_evidence)
    if not (
        smoke.get("status") == "PASS"
        and smoke.get("scientific") is False
        and smoke.get("synthetic_unprotected_data_only") is True
        and smoke.get("source_git_sha") == args.authorized_source_sha
        and smoke.get("restore_success") is True
        and smoke.get("resume_success") is True
    ):
        raise SystemExit("real G1 sealing requires a green real-Kaggle non-scientific infrastructure smoke attestation")

    require_sha256(args.manifest, MANIFEST_SHA256, "V1 manifest")
    require_sha256(args.class_map, CLASS_MAP_SHA256, "class map")
    require_sha256(args.teacher_checkpoint, TEACHER_SHA256, "historical teacher")

    dependency = _load(repo / args.dependency_lock)
    dep_errors = validate_dependency_lock_object(dependency) + validate_dependency_environment(dependency)
    if dep_errors:
        raise SystemExit("dependency lock validation failed: " + "; ".join(dep_errors))

    provenance = _load(args.pretrained_provenance)
    prov_errors = validate_pretrained_provenance(provenance, artifact_path=args.pretrained)
    if prov_errors:
        raise SystemExit("MobileNetV4 provenance validation failed: " + "; ".join(prov_errors))

    factory = _load(args.teacher_factory_manifest)
    factory_errors = validate_teacher_factory_bundle(
        factory, source_root=(args.teacher_factory_root or repo), expected_entrypoint=args.teacher_factory
    )
    if factory_errors:
        raise SystemExit("teacher factory bundle validation failed: " + "; ".join(factory_errors))

    class_order = _load(args.teacher_class_order_evidence)
    order_errors = validate_teacher_class_order_evidence(
        class_order,
        factory_bundle_sha256=factory.get("bundle_sha256"),
    )
    if order_errors:
        raise SystemExit("teacher class-order validation failed: " + "; ".join(order_errors))

    private_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    pretrained_sha = provenance["artifact_sha256"]
    pair_rows = {}
    for key, spec in PAIR_SPECS.items():
        binary = private_dir / f"PAIR_INIT_{key}.pt"
        model = create_student_from_pretrained(args.pretrained, seed=spec["seed"], num_classes=120)
        init_sha = save_pair_initialization(
            model,
            binary,
            pair_id=spec["pair_id"],
            seed=spec["seed"],
            pretrained_sha256=pretrained_sha,
        )
        ev = {
            "schema_version": "2.0",
            "pair_id": spec["pair_id"],
            "seed": spec["seed"],
            "model_name": MNV4_MODEL_NAME,
            "num_classes": 120,
            "pretrained_sha256": pretrained_sha,
            "student_init_sha256": init_sha,
            "student_init_bytes": binary.stat().st_size,
            "student_init_basename": binary.name,
            "authorized_consumers": spec["consumers"],
            "generated_under_source_sha": args.authorized_source_sha,
        }
        ev_path = evidence_dir / f"PAIR_INIT_{key}.json"
        atomic_write_json(ev_path, ev)
        pair_rows[key] = {
            "pair_id": spec["pair_id"],
            "seed": spec["seed"],
            "authorized_consumers": spec["consumers"],
            "pretrained_sha256": pretrained_sha,
            "sha256": init_sha,
            "bytes": binary.stat().st_size,
            "basename": binary.name,
            "evidence_sha256": sha256_json(ev),
        }

    teacher_byte_evidence = {
        "schema_version": "2.0",
        "kind": "teacher",
        "sha256": TEACHER_SHA256,
        "bytes": Path(args.teacher_checkpoint).stat().st_size,
        "artifact_basename": Path(args.teacher_checkpoint).name,
        "verification": "exact_byte_hash",
        "class_map_sha256": CLASS_MAP_SHA256,
    }

    # Persist the evidence objects used by the seal; no restricted model bytes are copied into public Git.
    atomic_write_json(evidence_dir / "MNV4_PRETRAINED_PROVENANCE.json", provenance)
    atomic_write_json(evidence_dir / "TEACHER_FACTORY_BUNDLE.json", factory)
    atomic_write_json(evidence_dir / "DINO_TEACHER.json", teacher_byte_evidence)
    atomic_write_json(evidence_dir / "TEACHER_CLASS_ORDER_EVIDENCE.json", class_order)

    seal = {
        "schema_version": "1.0",
        "authority": {"id": AUTHORITY_ID, "sha256": AUTHORITY_SHA256},
        "source_git_sha": args.authorized_source_sha,
        "dataset": {"manifest_sha256": MANIFEST_SHA256, "class_map_sha256": CLASS_MAP_SHA256},
        "student": {
            "model_name": MNV4_MODEL_NAME,
            "timm_version": TIMM_VERSION,
            "pretrained": {
                "sha256": pretrained_sha,
                "bytes": int(provenance["artifact_bytes"]),
                "basename": provenance["artifact_basename"],
                "source_kind": provenance["source_kind"],
                "source_locator": provenance["source_locator"],
                "timm_pretrained_cfg_sha256": provenance["timm_pretrained_cfg_sha256"],
                "provenance_sha256": sha256_json(provenance),
            },
        },
        "pair_initializations": pair_rows,
        "teacher": {
            "checkpoint_sha256": TEACHER_SHA256,
            "checkpoint_bytes": Path(args.teacher_checkpoint).stat().st_size,
            "class_map_sha256": CLASS_MAP_SHA256,
            "byte_evidence_sha256": sha256_json(teacher_byte_evidence),
            "factory_entrypoint": args.teacher_factory,
            "factory_bundle_sha256": factory["bundle_sha256"],
            "factory_manifest_sha256": sha256_json(factory),
            "class_order_evidence_sha256": sha256_json(class_order),
        },
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "infra_smoke_evidence_sha256": sha256_json(smoke),
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    seal["g1_seal_sha256"] = g1_seal_hash(seal)
    atomic_write_json(seal_path, seal)
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
