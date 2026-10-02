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
    validate_dependency_environment,
    validate_dependency_lock_object,
    validate_g1_seal_object,
)
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.secondary import (
    AUTHORITY_ID,
    AUTHORITY_SHA256,
    BASELINE_SPECS,
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    PRINCIPAL_SCIENCE_SOURCE_SHA,
    PRINCIPAL_G1_SEAL_SHA256,
    PRINCIPAL_S1_INIT_SHA256,
    PRINCIPAL_MNV4_PRETRAINED_SHA256,
    S1_PAIR_ID,
    S1_SEED,
    TEACHER_SHA256,
    TORCHVISION_VERSION,
    create_baseline_from_pretrained,
    save_baseline_initialization,
    secondary_g1_hash,
    validate_secondary_g1_bundle,
    validate_torchvision_provenance,
)
from cropcop_je.smoke_handoff import (
    validate_terminal_dual_gpu_smoke_evidence,
    validate_terminal_smoke_b_evidence,
)
from cropcop_je.source_state import verify_clean_source


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_exact(source: str | Path, target: Path, *, expected_sha256: str, label: str) -> None:
    source = Path(source)
    require_sha256(source, expected_sha256, label)
    if target.exists():
        raise SystemExit(f"refuse to overwrite existing sealed {label}: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require_sha256(target, expected_sha256, f"sealed {label}")
    if target.stat().st_size != source.stat().st_size:
        raise SystemExit(f"sealed {label} byte count changed during copy")


def verify_principal_anchor(bundle: Path) -> tuple[dict, dict]:
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file():
        raise SystemExit("principal G1 bundle lacks G1_MODEL_IDENTITY_SEAL.json")
    seal = load(seal_path)
    errors = validate_g1_seal_object(seal)
    if errors:
        raise SystemExit("principal G1 seal invalid: " + "; ".join(errors))
    if seal.get("source_git_sha") != PRINCIPAL_SCIENCE_SOURCE_SHA:
        raise SystemExit("principal G1 does not bind frozen principal source f171309")
    if seal.get("g1_seal_sha256") != PRINCIPAL_G1_SEAL_SHA256:
        raise SystemExit("principal G1 exact seal SHA differs from terminal principal evidence")

    pair = seal.get("pair_initializations", {}).get("S1", {})
    if pair.get("pair_id") != S1_PAIR_ID or int(pair.get("seed", -1)) != S1_SEED:
        raise SystemExit("principal G1 S1 pair identity mismatch")
    if pair.get("sha256") != PRINCIPAL_S1_INIT_SHA256:
        raise SystemExit("principal G1 S1 initialization SHA differs from terminal principal evidence")
    if pair.get("pretrained_sha256") != PRINCIPAL_MNV4_PRETRAINED_SHA256:
        raise SystemExit("principal G1 MobileNetV4 pretrained SHA differs from terminal principal evidence")
    pair_binary = bundle / "private" / str(pair.get("basename", ""))
    require_sha256(pair_binary, pair.get("sha256", ""), "principal S1 pair initialization")
    pair_ev_path = bundle / "evidence" / "PAIR_INIT_S1.json"
    pair_ev = load(pair_ev_path)
    if sha256_json(pair_ev) != pair.get("evidence_sha256"):
        raise SystemExit("principal S1 pair evidence differs from principal G1 seal")
    if pair_ev.get("student_init_sha256") != pair.get("sha256"):
        raise SystemExit("principal S1 pair evidence/binary SHA mismatch")

    teacher = seal.get("teacher", {})
    teacher_binary = bundle / "private" / str(teacher.get("artifact_basename", ""))
    require_sha256(teacher_binary, TEACHER_SHA256, "principal historical teacher")
    checks = (
        ("TEACHER_FACTORY_BUNDLE.json", "factory_manifest_sha256"),
        ("TEACHER_CLASS_ORDER_EVIDENCE.json", "class_order_evidence_sha256"),
        ("TEACHER_CANONICAL_STATE_EVIDENCE.json", "canonical_state_evidence_sha256"),
        ("TEACHER_ADAPTER_PARITY_EVIDENCE.json", "adapter_parity_evidence_sha256"),
    )
    for basename, field in checks:
        obj = load(bundle / "evidence" / basename)
        if sha256_json(obj) != teacher.get(field):
            raise SystemExit(f"principal teacher evidence differs from G1 seal: {basename}")
    return seal, pair_ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--principal-g1-bundle", required=True)
    ap.add_argument("--effb0-pretrained", required=True)
    ap.add_argument("--effb0-provenance", required=True)
    ap.add_argument("--cnxtt-pretrained", required=True)
    ap.add_argument("--cnxtt-provenance", required=True)
    ap.add_argument("--infra-smoke-evidence", required=True)
    ap.add_argument("--dual-gpu-smoke-evidence", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--bundle-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.bundle_dir).resolve()
    if (bundle / "SECONDARY_G1_MODEL_IDENTITY_SEAL.json").exists():
        raise SystemExit("secondary G1 target already contains a seal; immutable G1 cannot be overwritten")
    verify_clean_source(repo, authorized_source_sha=args.authorized_source_sha, output_roots=[bundle])

    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = load(dep_path)
    dep_errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if dep_errors:
        raise SystemExit("dependency lock validation failed: " + "; ".join(dep_errors))

    smoke = load(args.infra_smoke_evidence)
    smoke_errors = validate_terminal_smoke_b_evidence(
        smoke,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dep["dependency_lock_sha256"],
        require_batch=True,
    )
    if smoke_errors:
        raise SystemExit("secondary G1 requires source-qualified terminal Smoke-B: " + "; ".join(smoke_errors))
    smoke_sha = sha256_json(smoke)

    dual = load(args.dual_gpu_smoke_evidence)
    dual_errors = validate_terminal_dual_gpu_smoke_evidence(
        dual,
        expected_source_sha=args.authorized_source_sha,
        expected_dependency_lock_sha256=dep["dependency_lock_sha256"],
        expected_amendment_id=AMENDMENT_ID,
        expected_amendment_sha256=AMENDMENT_SHA256,
        expected_smoke_b_evidence_sha256=smoke_sha,
        require_batch=True,
    )
    if dual_errors:
        raise SystemExit("secondary G1 requires source-qualified dual-GPU smoke: " + "; ".join(dual_errors))
    dual_sha = sha256_json(dual)

    require_sha256(args.manifest, MANIFEST_SHA256, "frozen V1 manifest")
    require_sha256(args.class_map, CLASS_MAP_SHA256, "frozen 120-way class map")

    principal_root = Path(args.principal_g1_bundle).resolve()
    principal_seal, principal_pair_ev = verify_principal_anchor(principal_root)
    principal_pair = principal_seal["pair_initializations"]["S1"]
    principal_teacher = principal_seal["teacher"]

    baseline_inputs = {
        "effb0": (Path(args.effb0_pretrained), Path(args.effb0_provenance)),
        "cnxtt": (Path(args.cnxtt_pretrained), Path(args.cnxtt_provenance)),
    }
    provenance = {}
    for key, (artifact, prov_path) in baseline_inputs.items():
        prov = load(prov_path)
        errors = validate_torchvision_provenance(prov, model_key=key, artifact_path=artifact)
        if errors:
            raise SystemExit(f"{key} pretrained provenance invalid: " + "; ".join(errors))
        provenance[key] = prov

    private = bundle / "private"
    evidence = bundle / "evidence"
    private.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)

    pair_target = private / "PAIR_INIT_S1.pt"
    copy_exact(
        principal_root / "private" / principal_pair["basename"],
        pair_target,
        expected_sha256=principal_pair["sha256"],
        label="frozen MobileNetV4 S1 initialization",
    )
    secondary_pair_ev = {
        "schema_version": "3.0",
        "pair_id": S1_PAIR_ID,
        "seed": S1_SEED,
        "model_name": principal_pair_ev["model_name"],
        "num_classes": 120,
        "pretrained_sha256": principal_pair["pretrained_sha256"],
        "student_init_sha256": principal_pair["sha256"],
        "student_init_bytes": pair_target.stat().st_size,
        "student_init_basename": pair_target.name,
        "authorized_consumers": ["R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1"],
        "principal_pair_evidence_sha256": principal_pair["evidence_sha256"],
        "principal_g1_seal_sha256": principal_seal["g1_seal_sha256"],
        "principal_science_source_sha": PRINCIPAL_SCIENCE_SOURCE_SHA,
        "generated_under_source_sha": args.authorized_source_sha,
        "initialization_bytes_reused_exactly": True,
    }
    atomic_write_json(evidence / "PAIR_INIT_S1.json", secondary_pair_ev)

    teacher_target = private / "DINO_TEACHER.pt"
    copy_exact(
        principal_root / "private" / principal_teacher["artifact_basename"],
        teacher_target,
        expected_sha256=TEACHER_SHA256,
        label="historical DINO teacher",
    )
    for basename in (
        "MNV4_PRETRAINED_PROVENANCE.json",
        "DINO_TEACHER.json",
        "TEACHER_FACTORY_BUNDLE.json",
        "TEACHER_CLASS_ORDER_EVIDENCE.json",
        "TEACHER_CANONICAL_STATE_EVIDENCE.json",
        "TEACHER_ADAPTER_PARITY_EVIDENCE.json",
    ):
        src = principal_root / "evidence" / basename
        if not src.is_file():
            raise SystemExit(f"principal G1 evidence missing: {basename}")
        shutil.copyfile(src, evidence / basename)

    baseline_rows = {}
    for key, spec in BASELINE_SPECS.items():
        source_artifact, _source_prov = baseline_inputs[key]
        pretrained_target = private / spec["pretrained_basename"]
        copy_exact(source_artifact, pretrained_target, expected_sha256=provenance[key]["artifact_sha256"], label=f"{key} official pretrained")
        prov = dict(provenance[key])
        prov["sealed_basename"] = pretrained_target.name
        prov["sealed_sha256"] = sha256_file(pretrained_target)
        prov["sealed_bytes"] = pretrained_target.stat().st_size
        atomic_write_json(evidence / spec["pretrained_evidence_basename"], prov)

        model = create_baseline_from_pretrained(pretrained_target, model_key=key, seed=S1_SEED, num_classes=120)
        init_target = private / spec["init_basename"]
        init_sha = save_baseline_initialization(
            model,
            init_target,
            model_key=key,
            seed=S1_SEED,
            pretrained_sha256=prov["artifact_sha256"],
            authorized_consumers=[spec["consumer"]],
        )
        init_ev = {
            "schema_version": "1.0",
            "model_key": key,
            "model_name": spec["model_name"],
            "torchvision_version": TORCHVISION_VERSION,
            "weight_enum": spec["weight_enum"],
            "seed": S1_SEED,
            "num_classes": 120,
            "pretrained_sha256": prov["artifact_sha256"],
            "init_sha256": init_sha,
            "init_bytes": init_target.stat().st_size,
            "init_basename": init_target.name,
            "authorized_consumers": [spec["consumer"]],
            "generated_under_source_sha": args.authorized_source_sha,
        }
        atomic_write_json(evidence / spec["init_evidence_basename"], init_ev)
        baseline_rows[key] = {
            "model_name": spec["model_name"],
            "torchvision_version": TORCHVISION_VERSION,
            "weight_enum": spec["weight_enum"],
            "official_filename": spec["official_filename"],
            "pretrained_sha256": prov["artifact_sha256"],
            "pretrained_bytes": pretrained_target.stat().st_size,
            "pretrained_basename": pretrained_target.name,
            "pretrained_provenance_sha256": sha256_json(prov),
            "init_sha256": init_sha,
            "init_bytes": init_target.stat().st_size,
            "init_basename": init_target.name,
            "init_evidence_sha256": sha256_json(init_ev),
            "seed": S1_SEED,
            "authorized_consumers": [spec["consumer"]],
        }

    teacher_bundle = load(evidence / "TEACHER_FACTORY_BUNDLE.json")
    teacher_class = load(evidence / "TEACHER_CLASS_ORDER_EVIDENCE.json")
    teacher_state = load(evidence / "TEACHER_CANONICAL_STATE_EVIDENCE.json")
    teacher_adapter = load(evidence / "TEACHER_ADAPTER_PARITY_EVIDENCE.json")

    seal = {
        "schema_version": "1.0",
        "authority": {"id": AUTHORITY_ID, "sha256": AUTHORITY_SHA256},
        "source_git_sha": args.authorized_source_sha,
        "principal_science_source_sha": PRINCIPAL_SCIENCE_SOURCE_SHA,
        "principal_g1_seal_sha256": principal_seal["g1_seal_sha256"],
        "dataset": {"manifest_sha256": MANIFEST_SHA256, "class_map_sha256": CLASS_MAP_SHA256, "v1_test_accessed": False},
        "mnv4_s1": {
            "pair_id": S1_PAIR_ID,
            "seed": S1_SEED,
            "student_init_sha256": principal_pair["sha256"],
            "student_init_bytes": pair_target.stat().st_size,
            "student_init_basename": pair_target.name,
            "student_init_evidence_sha256": sha256_json(secondary_pair_ev),
            "pretrained_sha256": principal_pair["pretrained_sha256"],
            "pretrained_provenance_sha256": sha256_json(load(evidence / "MNV4_PRETRAINED_PROVENANCE.json")),
            "authorized_consumers": ["R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1"],
            "initialization_bytes_reused_exactly": True,
        },
        "teacher": {
            "checkpoint_sha256": TEACHER_SHA256,
            "checkpoint_bytes": teacher_target.stat().st_size,
            "artifact_basename": teacher_target.name,
            "factory_entrypoint": principal_teacher["factory_entrypoint"],
            "factory_bundle_sha256": teacher_bundle["bundle_sha256"],
            "factory_manifest_sha256": sha256_json(teacher_bundle),
            "class_order_evidence_sha256": sha256_json(teacher_class),
            "canonical_state_evidence_sha256": sha256_json(teacher_state),
            "adapter_parity_evidence_sha256": sha256_json(teacher_adapter),
            "reused_exactly_from_principal_g1": True,
        },
        "baselines": baseline_rows,
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "infra_smoke_evidence_sha256": smoke_sha,
        "dual_gpu_smoke_evidence_sha256": dual_sha,
        "stage04_execution_amendment": {"id": AMENDMENT_ID, "sha256": AMENDMENT_SHA256},
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    seal["secondary_g1_seal_sha256"] = secondary_g1_hash(seal)
    atomic_write_json(bundle / "SECONDARY_G1_MODEL_IDENTITY_SEAL.json", seal)

    _verified, errors = validate_secondary_g1_bundle(bundle)
    if errors:
        raise SystemExit("secondary G1 self-validation failed: " + "; ".join(errors))
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
