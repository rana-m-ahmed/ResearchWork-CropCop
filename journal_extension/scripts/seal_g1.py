from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.envelope import AMENDMENT_ID, AMENDMENT_SHA256
from cropcop_je.g1 import (
    AUTHORITY_ID,
    AUTHORITY_SHA256,
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    MNV4_MODEL_NAME,
    PAIR_SPECS,
    TEACHER_SHA256,
    TIMM_VERSION,
    assert_g1_creation_target_fresh,
    g1_seal_hash,
    validate_dependency_environment,
    validate_dependency_lock_object,
    validate_pretrained_provenance,
    validate_teacher_canonical_state_evidence,
    validate_teacher_class_order_evidence,
    validate_teacher_factory_bundle,
)
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.models import create_student_from_pretrained, save_pair_initialization
from cropcop_je.smoke_handoff import (
    validate_terminal_dual_gpu_smoke_evidence,
    validate_terminal_smoke_b_evidence,
)
from cropcop_je.source_state import verify_clean_source

SEMANTIC_MANIFEST_FINGERPRINT = (
    "7c368e6e3d8be3bb3a9a3a5f961075d4faa125bcac2e98a3b55e1a1c61f1c523"
)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _copy_exact(source: str | Path, target: Path, *, expected_sha256: str, label: str) -> str:
    source = Path(source)
    require_sha256(source, expected_sha256, label)
    if target.exists():
        raise SystemExit(f"refuse to overwrite existing sealed {label}: {target}")
    shutil.copyfile(source, target)
    observed = require_sha256(target, expected_sha256, f"sealed {label} copy")
    if target.stat().st_size != source.stat().st_size:
        raise SystemExit(f"sealed {label} byte count changed during copy")
    return observed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--pretrained-provenance", required=True)
    ap.add_argument("--pretrained-preparation", default="")
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--teacher-factory", required=True)
    ap.add_argument("--teacher-factory-manifest", required=True)
    ap.add_argument("--teacher-factory-root", default="")
    ap.add_argument("--teacher-class-order-evidence", required=True)
    ap.add_argument("--teacher-canonical-state-evidence", required=True)
    ap.add_argument("--teacher-adapter-parity-evidence", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--infra-smoke-evidence", required=True)
    ap.add_argument("--dual-gpu-smoke-evidence", required=True)
    ap.add_argument("--bundle-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.bundle_dir).resolve()
    verify_clean_source(repo, authorized_source_sha=args.authorized_source_sha, output_roots=[bundle])

    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    private_dir = bundle / "private"
    evidence_dir = bundle / "evidence"
    assert_g1_creation_target_fresh(bundle)

    dependency = _load(repo / args.dependency_lock)
    dep_errors = validate_dependency_lock_object(dependency) + validate_dependency_environment(dependency)
    if dep_errors:
        raise SystemExit("dependency lock validation failed: " + "; ".join(dep_errors))

    smoke = _load(args.infra_smoke_evidence)
    smoke_errors = validate_terminal_smoke_b_evidence(
        smoke,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        require_batch=True,
    )
    if smoke_errors:
        raise SystemExit(
            "real G1 sealing requires a green real-Kaggle canonical terminal Batch Smoke-B evidence: "
            + "; ".join(smoke_errors)
        )
    smoke_sha = sha256_json(smoke)

    dual_smoke = _load(args.dual_gpu_smoke_evidence)
    dual_errors = validate_terminal_dual_gpu_smoke_evidence(
        dual_smoke,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        expected_amendment_id=AMENDMENT_ID,
        expected_amendment_sha256=AMENDMENT_SHA256,
        expected_smoke_b_evidence_sha256=smoke_sha,
        require_batch=True,
    )
    if dual_errors:
        raise SystemExit(
            "real G1 sealing requires green terminal dual-GPU-smoke evidence: "
            + "; ".join(dual_errors)
        )
    dual_smoke_sha = sha256_json(dual_smoke)

    require_sha256(args.manifest, MANIFEST_SHA256, "V1 manifest")
    require_sha256(args.class_map, CLASS_MAP_SHA256, "class map")
    require_sha256(args.teacher_checkpoint, TEACHER_SHA256, "historical teacher")

    provenance = _load(args.pretrained_provenance)
    prov_errors = validate_pretrained_provenance(provenance, artifact_path=args.pretrained)
    if prov_errors:
        raise SystemExit("MobileNetV4 provenance validation failed: " + "; ".join(prov_errors))

    factory = _load(args.teacher_factory_manifest)
    factory_errors = validate_teacher_factory_bundle(
        factory,
        source_root=(args.teacher_factory_root or repo),
        expected_entrypoint=args.teacher_factory,
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

    canonical_state = _load(args.teacher_canonical_state_evidence)
    canonical_errors = validate_teacher_canonical_state_evidence(
        canonical_state,
        factory_bundle_sha256=factory.get("bundle_sha256"),
    )
    if canonical_errors:
        raise SystemExit("teacher canonical-state validation failed: " + "; ".join(canonical_errors))

    adapter_parity = _load(args.teacher_adapter_parity_evidence)
    if adapter_parity.get("status") != "PASS" or adapter_parity.get("direct_vs_adapter_parity") is not True:
        raise SystemExit("teacher synthetic direct-vs-adapter parity is not PASS")
    if adapter_parity.get("protected_data_accessed") is not False:
        raise SystemExit("teacher adapter parity evidence accessed protected data")
    if adapter_parity.get("scientific_metric_computed") is not False:
        raise SystemExit("teacher adapter parity evidence computed a scientific metric")

    private_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)

    pretrained_source = Path(args.pretrained)
    suffix = pretrained_source.suffix if pretrained_source.suffix else ".pt"
    sealed_pretrained = private_dir / f"MNV4_PRETRAINED{suffix}"
    pretrained_sha = provenance["artifact_sha256"]
    _copy_exact(
        pretrained_source,
        sealed_pretrained,
        expected_sha256=pretrained_sha,
        label="MobileNetV4 pretrained candidate",
    )

    sealed_teacher = private_dir / "DINO_TEACHER.pt"
    _copy_exact(
        args.teacher_checkpoint,
        sealed_teacher,
        expected_sha256=TEACHER_SHA256,
        label="historical DINO teacher",
    )

    internal_provenance = dict(provenance)
    internal_provenance["source_candidate_basename"] = provenance["artifact_basename"]
    internal_provenance["artifact_basename"] = sealed_pretrained.name
    internal_provenance["sealed_copy_sha256"] = sha256_file(sealed_pretrained)
    internal_provenance["sealed_copy_bytes"] = sealed_pretrained.stat().st_size
    internal_provenance["sealed_copy_verification"] = "exact_byte_copy"

    pair_rows = {}
    for key, spec in PAIR_SPECS.items():
        binary = private_dir / f"PAIR_INIT_{key}.pt"
        model = create_student_from_pretrained(sealed_pretrained, seed=spec["seed"], num_classes=120)
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
        atomic_write_json(evidence_dir / f"PAIR_INIT_{key}.json", ev)
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
        "bytes": sealed_teacher.stat().st_size,
        "artifact_basename": sealed_teacher.name,
        "verification": "exact_byte_hash_and_sealed_copy",
        "class_map_sha256": CLASS_MAP_SHA256,
        "canonical_state": "EMA",
    }
    frozen_v1_identity = {
        "schema_version": "1.0",
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "semantic_manifest_fingerprint": SEMANTIC_MANIFEST_FINGERPRINT,
        "protected_test_accessed_during_g1": False,
    }

    atomic_write_json(evidence_dir / "MNV4_PRETRAINED_PROVENANCE.json", internal_provenance)
    if args.pretrained_preparation:
        atomic_write_json(evidence_dir / "MNV4_PRETRAINED_PREPARATION.json", _load(args.pretrained_preparation))
    atomic_write_json(evidence_dir / "TEACHER_FACTORY_BUNDLE.json", factory)
    atomic_write_json(evidence_dir / "DINO_TEACHER.json", teacher_byte_evidence)
    atomic_write_json(evidence_dir / "TEACHER_CLASS_ORDER_EVIDENCE.json", class_order)
    atomic_write_json(evidence_dir / "TEACHER_CANONICAL_STATE_EVIDENCE.json", canonical_state)
    atomic_write_json(evidence_dir / "TEACHER_ADAPTER_PARITY_EVIDENCE.json", adapter_parity)
    atomic_write_json(evidence_dir / "FROZEN_V1_IDENTITY.json", frozen_v1_identity)

    seal = {
        "schema_version": "2.0",
        "authority": {"id": AUTHORITY_ID, "sha256": AUTHORITY_SHA256},
        "source_git_sha": args.authorized_source_sha,
        "dataset": {
            "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256,
            "semantic_manifest_fingerprint": SEMANTIC_MANIFEST_FINGERPRINT,
            "identity_evidence_sha256": sha256_json(frozen_v1_identity),
        },
        "student": {
            "model_name": MNV4_MODEL_NAME,
            "timm_version": TIMM_VERSION,
            "pretrained": {
                "sha256": pretrained_sha,
                "bytes": sealed_pretrained.stat().st_size,
                "basename": sealed_pretrained.name,
                "source_kind": provenance["source_kind"],
                "source_locator": provenance["source_locator"],
                "timm_pretrained_cfg_sha256": provenance["timm_pretrained_cfg_sha256"],
                "tensor_identity_sha256": provenance["tensor_identity_sha256"],
                "candidate_serialization_format": provenance["candidate_serialization_format"],
                "provenance_sha256": sha256_json(internal_provenance),
            },
        },
        "pair_initializations": pair_rows,
        "teacher": {
            "checkpoint_sha256": TEACHER_SHA256,
            "checkpoint_bytes": sealed_teacher.stat().st_size,
            "artifact_basename": sealed_teacher.name,
            "class_map_sha256": CLASS_MAP_SHA256,
            "canonical_state": "EMA",
            "byte_evidence_sha256": sha256_json(teacher_byte_evidence),
            "factory_entrypoint": args.teacher_factory,
            "factory_bundle_sha256": factory["bundle_sha256"],
            "factory_manifest_sha256": sha256_json(factory),
            "class_order_evidence_sha256": sha256_json(class_order),
            "canonical_state_evidence_sha256": sha256_json(canonical_state),
            "adapter_parity_evidence_sha256": sha256_json(adapter_parity),
        },
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "infra_smoke_evidence_sha256": smoke_sha,
        "dual_gpu_smoke_evidence_sha256": dual_smoke_sha,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    seal["g1_seal_sha256"] = g1_seal_hash(seal)
    atomic_write_json(seal_path, seal)
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
