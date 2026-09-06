from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "journal_extension" / "scripts"
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import (
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    validate_dependency_environment,
    validate_dependency_lock_object,
)
from cropcop_je.g1_inputs import input_identity_summary, resolve_creation_inputs
from cropcop_je.g1_package import readiness_transport_dry_run
from cropcop_je.g1_publication import readiness_private_target_probe
from cropcop_je.hashing import sha256_file
from cropcop_je.source_state import verify_clean_source


DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/cropcop-g1-readiness")


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required G1 readiness environment variable missing: {name}")
    return value


def execute(script: str, args: list[str]) -> None:
    print(f"+ readiness execute {script} [arguments redacted]")
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT, check=True)


def _fresh_output_root() -> Path:
    raw = os.environ.get("CROPCOP_G1_READINESS_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)).strip()
    root = Path(raw).expanduser().resolve()
    if root == ROOT or ROOT in root.parents:
        raise RuntimeError("G1 readiness output root must remain outside the Git checkout")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"G1 readiness output root must be fresh: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> int:
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    if len(source_sha) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in source_sha):
        raise RuntimeError("CROPCOP_SOURCE_GIT_COMMIT must be an immutable 40-hex commit")

    output_root = _fresh_output_root()
    verify_clean_source(
        ROOT,
        authorized_source_sha=source_sha,
        output_roots=[output_root],
    )

    dependency_path = ROOT / "journal_extension/locks/execution_dependency_lock.json"
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    dependency_errors = validate_dependency_lock_object(dependency)
    dependency_errors.extend(validate_dependency_environment(dependency))
    if dependency_errors:
        raise RuntimeError(
            "G1 readiness dependency environment invalid: " + "; ".join(dependency_errors)
        )

    inputs = resolve_creation_inputs(ROOT, os.environ)
    private_target = readiness_private_target_probe(
        inputs.private_dataset_slug,
        env=os.environ,
    )

    prep_root = output_root / "prep"
    evidence_root = output_root / "evidence"
    staging_root = output_root / "package-dry-run-staging"
    private_root = staging_root / "private"
    staged_evidence = staging_root / "evidence"
    for path in (prep_root, evidence_root, private_root, staged_evidence):
        path.mkdir(parents=True, exist_ok=False)

    prep_record = evidence_root / "MNV4_PRETRAINED_PREPARATION.json"
    if inputs.mnv4_pretrained is None:
        execute("prepare_mnv4_pretrained.py", [
            "--output-dir", str(prep_root / "mnv4"),
            "--record", str(prep_record),
        ])
        prep_obj = json.loads(prep_record.read_text(encoding="utf-8"))
        mnv4 = Path(prep_obj["candidate_path"]).resolve()
    else:
        mnv4 = inputs.mnv4_pretrained
        atomic_write_json(prep_record, {
            "schema_version": "1.0",
            "status": "EXPLICIT_CANDIDATE_SUPPLIED",
            "candidate_path": str(mnv4),
            "candidate_sha256": sha256_file(mnv4),
            "candidate_bytes": mnv4.stat().st_size,
            "scientific_result_produced": False,
        })

    provenance = evidence_root / "MNV4_PRETRAINED_PROVENANCE.json"
    factory_manifest = evidence_root / "TEACHER_FACTORY_BUNDLE.json"
    canonical_state = evidence_root / "TEACHER_CANONICAL_STATE_EVIDENCE.json"
    parity = evidence_root / "TEACHER_ADAPTER_PARITY_EVIDENCE.json"
    class_order = evidence_root / "TEACHER_CLASS_ORDER_EVIDENCE.json"
    frozen_v1 = evidence_root / "FROZEN_V1_IDENTITY.json"

    execute("capture_mnv4_pretrained_provenance.py", [
        "--artifact", str(mnv4),
        "--output", str(provenance),
    ])
    execute("capture_teacher_factory_bundle.py", [
        "--repo-root", str(ROOT),
        "--source-root", str(inputs.teacher_factory_root),
        "--factory-spec", inputs.teacher_factory,
        "--output", str(factory_manifest),
    ])
    execute("capture_teacher_canonical_state.py", [
        "--checkpoint", str(inputs.teacher_checkpoint),
        "--teacher-factory-root", str(inputs.teacher_factory_root),
        "--teacher-factory", inputs.teacher_factory,
        "--teacher-factory-manifest", str(factory_manifest),
        "--output", str(canonical_state),
    ])
    execute("verify_teacher_adapter_parity.py", [
        "--checkpoint", str(inputs.teacher_checkpoint),
        "--teacher-factory-root", str(inputs.teacher_factory_root),
        "--teacher-factory", inputs.teacher_factory,
        "--teacher-factory-manifest", str(factory_manifest),
        "--output", str(parity),
    ])
    execute("verify_teacher_class_order.py", [
        "--repo-root", str(ROOT),
        "--class-map", str(inputs.class_map),
        "--teacher-checkpoint", str(inputs.teacher_checkpoint),
        "--teacher-factory", inputs.teacher_factory,
        "--teacher-factory-manifest", str(factory_manifest),
        "--teacher-factory-root", str(inputs.teacher_factory_root),
        "--historical-lineage-manifest", str(inputs.teacher_lineage_manifest),
        "--historical-evidence-root", str(inputs.teacher_historical_evidence_root),
        "--output", str(class_order),
    ])

    atomic_write_json(frozen_v1, {
        "schema_version": "1.0",
        "status": "PASS",
        "manifest_sha256": sha256_file(inputs.manifest),
        "class_map_sha256": sha256_file(inputs.class_map),
        "expected_manifest_sha256": MANIFEST_SHA256,
        "expected_class_map_sha256": CLASS_MAP_SHA256,
        "v1_test_accessed": False,
        "v1_validation_evaluated": False,
        "scientific_metric_computed": False,
    })

    # Exercise the actual deterministic tar-writing rules on the real verified
    # teacher/MNV4 bytes and the readiness evidence, without creating pair
    # initializations or a production G1 seal.
    shutil.copy2(inputs.teacher_checkpoint, private_root / "DINO_TEACHER.pt")
    shutil.copy2(mnv4, private_root / mnv4.name)
    for path in sorted(evidence_root.iterdir()):
        if path.is_file():
            shutil.copy2(path, staged_evidence / path.name)
    transport = readiness_transport_dry_run(
        staging_root,
        output_root / "transport-dry-run",
    )

    canonical_obj = json.loads(canonical_state.read_text(encoding="utf-8"))
    parity_obj = json.loads(parity.read_text(encoding="utf-8"))
    provenance_obj = json.loads(provenance.read_text(encoding="utf-8"))
    class_order_obj = json.loads(class_order.read_text(encoding="utf-8"))

    report = {
        "schema_version": "1.0",
        "status": "PASS",
        "source_git_sha": source_sha,
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "UNKNOWN"),
        "input_identity": input_identity_summary(inputs),
        "teacher_checkpoint_sha256": sha256_file(inputs.teacher_checkpoint),
        "teacher_checkpoint_bytes": inputs.teacher_checkpoint.stat().st_size,
        "teacher_factory_strict_real_checkpoint_load": canonical_obj.get("status") == "PASS",
        "teacher_complete_ema_overlay": canonical_obj.get("ema_exact_complete_coverage") is True,
        "teacher_canonical_tensor_equality": canonical_obj.get("canonical_floating_tensors_equal_ema") is True,
        "teacher_finite_state": canonical_obj.get("finite_state") is True,
        "teacher_adapter_parity": parity_obj.get("direct_vs_adapter_parity") is True,
        "teacher_adapter_max_absolute_difference": parity_obj.get("max_absolute_difference"),
        "class_order_lineage_verified": class_order_obj.get("status") == "PASS",
        "manifest_sha256": sha256_file(inputs.manifest),
        "class_map_sha256": sha256_file(inputs.class_map),
        "mnv4_official_tensor_match": provenance_obj.get("official_timm_tensor_match") is True,
        "mnv4_tensor_identity_sha256": provenance_obj.get("tensor_identity_sha256"),
        "mnv4_candidate_sha256": provenance_obj.get("artifact_sha256"),
        "private_target_preflight": private_target,
        "package_transport_dry_run": transport,
        "qualifying_phase": False,
        "g1_seal_created": False,
        "pair_initializations_created": False,
        "g1_publication_performed": False,
        "model_training_performed": False,
        "optimizer_steps_performed": 0,
        "v1_validation_evaluated": False,
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
        "scientific_metric_computed": False,
    }
    atomic_write_json(output_root / "G1_INPUT_READINESS.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
