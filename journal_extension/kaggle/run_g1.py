from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "journal_extension" / "scripts"
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.envelope import AMENDMENT_ID, AMENDMENT_SHA256
from cropcop_je.g1 import CLASS_MAP_SHA256, MANIFEST_SHA256
from cropcop_je.g1_barrier import (
    barrier_cli_args,
    from_environment as g1_barrier_from_environment,
)
from cropcop_je.g1_inputs import (
    CLASS_MAP_RELATIVE_PATH,
    DEFAULT_G1_BUNDLE_DIR,
    FINAL_MANIFEST_RELATIVE_PATH,
    resolve_creation_inputs,
)
from cropcop_je.g1_publication import (
    G1PublicationError,
    create_private_target_if_missing,
    preflight_private_target,
)
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.smoke_handoff import (
    require_qualifying_kaggle_batch,
    validate_terminal_dual_gpu_smoke_evidence,
    validate_terminal_smoke_b_evidence,
)


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required G1 environment variable missing: {name}")
    return value


def execute(script: str, args: list[str]) -> None:
    print(f"+ execute {script} [arguments redacted]")
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT, check=True)


def _set_final_v1_paths_from_root() -> tuple[Path, Path]:
    root = Path(req("CROPCOP_FINAL_V1_ROOT")).resolve()
    manifest = root / FINAL_MANIFEST_RELATIVE_PATH
    class_map = root / CLASS_MAP_RELATIVE_PATH
    require_sha256(manifest, MANIFEST_SHA256, "frozen V1 manifest")
    require_sha256(class_map, CLASS_MAP_SHA256, "frozen class map")
    os.environ["CROPCOP_MANIFEST"] = str(manifest)
    os.environ["CROPCOP_CLASS_MAP"] = str(class_map)
    return manifest, class_map


def _smoke_preflight(source_sha: str) -> tuple[str, str, dict]:
    smoke_path = req("CROPCOP_INFRA_SMOKE_EVIDENCE")
    dual_path = req("CROPCOP_DUAL_GPU_SMOKE_EVIDENCE")
    dependency = json.loads(
        (ROOT / "journal_extension/locks/execution_dependency_lock.json").read_text(encoding="utf-8")
    )
    smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
    errors = validate_terminal_smoke_b_evidence(
        smoke,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        require_batch=True,
    )
    if errors:
        raise RuntimeError("G1 preflight terminal Smoke-B invalid: " + "; ".join(errors))
    dual = json.loads(Path(dual_path).read_text(encoding="utf-8"))
    errors = validate_terminal_dual_gpu_smoke_evidence(
        dual,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        expected_amendment_id=AMENDMENT_ID,
        expected_amendment_sha256=AMENDMENT_SHA256,
        expected_smoke_b_evidence_sha256=sha256_json(smoke),
        require_batch=True,
    )
    if errors:
        raise RuntimeError("G1 preflight terminal dual-GPU-smoke invalid: " + "; ".join(errors))
    return smoke_path, dual_path, dependency


def _ensure_private_target(slug: str) -> dict:
    try:
        return preflight_private_target(slug, env=os.environ)
    except G1PublicationError:
        if os.environ.get("CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET", "").strip() != "1":
            raise
        create_private_target_if_missing(slug, env=os.environ)
        return preflight_private_target(slug, env=os.environ)


def _publish_terminal_evidence(source_sha: str, path: Path) -> str:
    return publish_to_github_branch(
        repo_dir=ROOT,
        source_git_sha=source_sha,
        run_id="G1",
        files=[path],
    )


def _publication_only_repair(
    source_sha: str,
    output_root: Path,
    smoke_path: str,
    dual_path: str,
    dependency: dict,
) -> int:
    _set_final_v1_paths_from_root()
    bundle = Path(os.environ.get("CROPCOP_G1_BUNDLE_DIR", str(DEFAULT_G1_BUNDLE_DIR))).resolve()
    os.environ["CROPCOP_G1_BUNDLE_DIR"] = str(bundle)
    os.environ["CROPCOP_INFRA_SMOKE_EVIDENCE"] = smoke_path
    os.environ["CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"] = dual_path
    inputs = g1_barrier_from_environment(ROOT, authorized_source_sha=source_sha, env=os.environ)
    barrier = output_root / "G1_BARRIER_REPAIR.json"
    execute("validate_g1_barrier.py", barrier_cli_args(inputs, output=barrier))
    slug = req("CROPCOP_G1_PRIVATE_DATASET_SLUG")
    _ensure_private_target(slug)
    receipt = output_root / "G1_PUBLICATION_RECEIPT.json"
    execute("publish_g1_bundle.py", [
        "--bundle-dir", str(bundle),
        "--dataset-slug", slug,
        "--receipt", str(receipt),
        "--mode", "repair",
    ])
    seal = json.loads((bundle / "G1_MODEL_IDENTITY_SEAL.json").read_text(encoding="utf-8"))
    receipt_obj = json.loads(receipt.read_text(encoding="utf-8"))
    terminal = {
        "schema_version": "1.0",
        "status": "PASS",
        "mode": "publication_repair",
        "source_git_sha": source_sha,
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "g1_seal_sha256": seal["g1_seal_sha256"],
        "package_sha256": receipt_obj["package_sha256"],
        "publication_receipt_sha256": sha256_file(receipt),
        "private_target_verified": True,
        "roundtrip_verified": True,
        "pair_initializations_regenerated": False,
        "seal_rewritten": False,
        "accelerator_required": False,
        "scientific_result_produced": False,
        "v1_test_accessed": False,
    }
    terminal_path = output_root / "G1_TERMINAL_EVIDENCE.json"
    atomic_write_json(terminal_path, terminal)
    branch = _publish_terminal_evidence(source_sha, terminal_path)
    terminal["public_evidence_branch"] = branch
    atomic_write_json(terminal_path, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0


def main() -> int:
    run_type = require_qualifying_kaggle_batch(context="G1 model-identity sealing")
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    if len(source_sha) != 40:
        raise RuntimeError("CROPCOP_SOURCE_GIT_COMMIT must be an immutable 40-hex source SHA")
    # Fail closed on source-bound qualification evidence before creating any
    # mutable G1 output or preparation directory.
    smoke_path, dual_path, dependency = _smoke_preflight(source_sha)

    output_root = Path(req("CROPCOP_OUTPUT_ROOT")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if os.environ.get("CROPCOP_G1_PUBLICATION_REPAIR", "").strip() == "1":
        return _publication_only_repair(
            source_sha,
            output_root,
            smoke_path,
            dual_path,
            dependency,
        )

    inputs = resolve_creation_inputs(ROOT, os.environ)
    os.environ["CROPCOP_MANIFEST"] = str(inputs.manifest)
    os.environ["CROPCOP_CLASS_MAP"] = str(inputs.class_map)
    os.environ["CROPCOP_G1_BUNDLE_DIR"] = str(inputs.g1_bundle_dir)
    os.environ["CROPCOP_INFRA_SMOKE_EVIDENCE"] = smoke_path
    os.environ["CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"] = dual_path

    target = _ensure_private_target(inputs.private_dataset_slug)
    prep = inputs.g1_prep_dir
    if prep.exists() and any(prep.iterdir()):
        raise RuntimeError(f"G1 preparation directory must be fresh: {prep}")
    prep.mkdir(parents=True, exist_ok=True)
    preseal = prep / "evidence"
    preseal.mkdir(parents=True, exist_ok=False)

    prep_record = preseal / "MNV4_PRETRAINED_PREPARATION.json"
    if inputs.mnv4_pretrained is None:
        execute("prepare_mnv4_pretrained.py", [
            "--output-dir", str(prep / "mnv4"),
            "--record", str(prep_record),
        ])
        prep_obj = json.loads(prep_record.read_text(encoding="utf-8"))
        pretrained = Path(prep_obj["candidate_path"]).resolve()
    else:
        pretrained = inputs.mnv4_pretrained
        atomic_write_json(prep_record, {
            "schema_version": "1.0",
            "status": "EXPLICIT_CANDIDATE_SUPPLIED",
            "candidate_path": str(pretrained),
            "candidate_sha256": sha256_file(pretrained),
            "scientific_result_produced": False,
        })

    provenance = preseal / "MNV4_PRETRAINED_PROVENANCE.json"
    factory_manifest = preseal / "TEACHER_FACTORY_BUNDLE.json"
    order_evidence = preseal / "TEACHER_CLASS_ORDER_EVIDENCE.json"
    canonical_evidence = preseal / "TEACHER_CANONICAL_STATE_EVIDENCE.json"
    parity_evidence = preseal / "TEACHER_ADAPTER_PARITY_EVIDENCE.json"

    execute("capture_mnv4_pretrained_provenance.py", [
        "--artifact", str(pretrained),
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
        "--output", str(canonical_evidence),
    ])
    execute("verify_teacher_adapter_parity.py", [
        "--checkpoint", str(inputs.teacher_checkpoint),
        "--teacher-factory-root", str(inputs.teacher_factory_root),
        "--teacher-factory", inputs.teacher_factory,
        "--teacher-factory-manifest", str(factory_manifest),
        "--output", str(parity_evidence),
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
        "--output", str(order_evidence),
    ])

    execute("seal_g1.py", [
        "--repo-root", str(ROOT),
        "--authorized-source-sha", source_sha,
        "--manifest", str(inputs.manifest),
        "--class-map", str(inputs.class_map),
        "--pretrained", str(pretrained),
        "--pretrained-provenance", str(provenance),
        "--pretrained-preparation", str(prep_record),
        "--teacher-checkpoint", str(inputs.teacher_checkpoint),
        "--teacher-factory", inputs.teacher_factory,
        "--teacher-factory-manifest", str(factory_manifest),
        "--teacher-factory-root", str(inputs.teacher_factory_root),
        "--teacher-class-order-evidence", str(order_evidence),
        "--teacher-canonical-state-evidence", str(canonical_evidence),
        "--teacher-adapter-parity-evidence", str(parity_evidence),
        "--infra-smoke-evidence", smoke_path,
        "--dual-gpu-smoke-evidence", dual_path,
        "--bundle-dir", str(inputs.g1_bundle_dir),
    ])

    barrier_inputs = g1_barrier_from_environment(
        ROOT,
        authorized_source_sha=source_sha,
        env=os.environ,
    )
    barrier = output_root / "G1_BARRIER.json"
    execute("validate_g1_barrier.py", barrier_cli_args(barrier_inputs, output=barrier))

    receipt = output_root / "G1_PUBLICATION_RECEIPT.json"
    execute("publish_g1_bundle.py", [
        "--bundle-dir", str(inputs.g1_bundle_dir),
        "--dataset-slug", inputs.private_dataset_slug,
        "--receipt", str(receipt),
        "--mode", "publish",
    ])

    seal = json.loads(
        (inputs.g1_bundle_dir / "G1_MODEL_IDENTITY_SEAL.json").read_text(encoding="utf-8")
    )
    receipt_obj = json.loads(receipt.read_text(encoding="utf-8"))
    try:
        import torch
        accelerator_observed = "CUDA_PRESENT_UNUSED" if torch.cuda.is_available() else "CPU"
    except Exception:
        accelerator_observed = "CPU"
    terminal = {
        "schema_version": "1.0",
        "status": "PASS",
        "kaggle_run_type": run_type,
        "source_git_sha": source_sha,
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "g1_seal_sha256": seal["g1_seal_sha256"],
        "smoke_b_evidence_sha256": seal["infra_smoke_evidence_sha256"],
        "dual_gpu_smoke_evidence_sha256": seal["dual_gpu_smoke_evidence_sha256"],
        "package_sha256": receipt_obj["package_sha256"],
        "publication_receipt_sha256": sha256_file(receipt),
        "private_target_verified": target["authoritative_is_private"],
        "roundtrip_verified": receipt_obj["roundtrip_verified"],
        "accelerator_required": False,
        "accelerator_observed": accelerator_observed,
        "scientific_result_produced": False,
        "model_training_performed": False,
        "v1_validation_evaluated": False,
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
    }
    terminal_path = output_root / "G1_TERMINAL_EVIDENCE.json"
    atomic_write_json(terminal_path, terminal)
    branch = _publish_terminal_evidence(source_sha, terminal_path)
    terminal["public_evidence_branch"] = branch
    atomic_write_json(terminal_path, terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
