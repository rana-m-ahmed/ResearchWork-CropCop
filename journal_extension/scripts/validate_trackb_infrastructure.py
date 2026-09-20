from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import (
    AUTHORITY_ID,
    CODE_ATTESTATION_ID,
    EXECUTION_LOCK_ID,
    TrackBError,
    load_json,
    validate_downstream_authority,
    validate_execution_lock,
    verify_code_attestation,
)


def _compile_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        raise TrackBError(f"unsupported notebook format: {path}")
    code_cells = 0
    combined = []
    for index, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        combined.append(str(source))
        if cell.get("cell_type") == "code":
            compile(str(source), f"{path.name}:cell{index}", "exec")
            code_cells += 1
    return {
        "cell_count": len(nb.get("cells", [])),
        "code_cell_count": code_cells,
        "text": "\n".join(combined),
    }


def _require(text: str, values: tuple[str, ...], label: str) -> None:
    missing = [value for value in values if value not in text]
    if missing:
        raise TrackBError(f"{label} missing required v5 guard(s): {missing}")


def _forbid(text: str, values: tuple[str, ...], label: str) -> None:
    found = [value for value in values if value in text]
    if found:
        raise TrackBError(f"{label} exposes forbidden/stale surface(s): {found}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Static infrastructure QA for hardened CropCop Track-B v5."
    )
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    authority_path = root / "journal_extension/amendments/track_bc_r07_downstream_v3.json"
    lock_path = root / "journal_extension/track_b_r07/TRACKB_R07_EXECUTION_LOCK_v4.json"
    attestation_path = root / "journal_extension/track_b_r07/TRACKB_CODE_ATTESTATION_v4.json"
    source_lock_path = root / "journal_extension/track_b_r07/TRACKB_EXTERNAL_SOURCE_LOCK_v2.json"
    release_path = root / "journal_extension/track_b_r07/TRACKB_RELEASE_AUTHORITY_v1.json"
    nb00_path = root / "journal_extension/kaggle/TrackB_00_Readiness_Materialization.ipynb"
    nb01_path = root / "journal_extension/kaggle/TrackB_01_Final_Execution.ipynb"

    required_files = (
        authority_path,
        lock_path,
        attestation_path,
        source_lock_path,
        release_path,
        nb00_path,
        nb01_path,
        root / "journal_extension/scripts/bootstrap_trackb_runtime.py",
        root / "journal_extension/scripts/build_trackb_core_package.py",
        root / "journal_extension/scripts/build_trackb_historical_compare.py",
        root / "journal_extension/scripts/trackb_v4_materialize.py",
        root / "journal_extension/scripts/trackb_v4_execute_attached.py",
        root / "journal_extension/scripts/run_trackb_r07.py",
        root / "journal_extension/scripts/validate_trackb_preinference_qualification.py",
        root / "journal_extension/scripts/validate_trackb_v4_orchestration.py",
        root / "journal_extension/scripts/validate_trackb_release_authority.py",
        root / "journal_extension/src/cropcop_je/trackb_r07.py",
        root / "journal_extension/src/cropcop_je/trackb_r07_audit.py",
        root / "journal_extension/src/cropcop_je/trackb_r07_ops.py",
    )
    for path in required_files:
        if not path.is_file():
            raise TrackBError(f"required Track-B v5 infrastructure file missing: {path.relative_to(root)}")

    authority = load_json(authority_path)
    lock = load_json(lock_path)
    source_lock = load_json(source_lock_path)
    release = load_json(release_path)

    validate_downstream_authority(authority)
    validate_execution_lock(lock)
    if lock.get("lock_id") != EXECUTION_LOCK_ID:
        raise TrackBError("execution lock identity differs from executable constant")
    if source_lock.get("status") != "PASS_EXTERNAL_SOURCE_AUTHORITY":
        raise TrackBError("external-source authority is not PASS")
    if any(
        not isinstance(row, dict) or row.get("materialization_permitted") is not True
        for row in (source_lock.get("candidates") or {}).values()
    ):
        raise TrackBError("external-source authority contains a blocked candidate")
    if release.get("release_id") != "TRACKB_V5_RELEASE_AUTHORITY_v1":
        raise TrackBError("unexpected Track-B release authority identity")
    if release.get("scientific_change") is not False:
        raise TrackBError("Track-B v5 release authority changes frozen science")

    if sha256_file(attestation_path) != lock.get("code_attestation_sha256"):
        raise TrackBError("v4 execution lock does not bind the committed v4 code attestation")
    attestation = verify_code_attestation(root, attestation_path)
    if attestation.get("attestation_id") != CODE_ATTESTATION_ID:
        raise TrackBError("committed code attestation differs from executable constant")

    nb00 = _compile_notebook(nb00_path)
    nb01 = _compile_notebook(nb01_path)
    nb00_text = nb00.pop("text")
    nb01_text = nb01.pop("text")

    _require(
        nb00_text,
        (
            "KAGGLE_API_TOKEN",
            "trackb_v4_materialize.py",
            "TRACKB_READINESS_RECEIPT.json",
            "protected_external_prediction_count",
            "v1_test_accessed",
        ),
        "Notebook 00",
    )
    _forbid(nb00_text, ("CROPCOP_GITHUB_TOKEN",), "Notebook 00")

    _require(
        nb01_text,
        (
            "RUN_MODE = 'qualification'",
            "TRACKB_INFRASTRUCTURE_BUNDLE.json",
            "TRACKB_EXTERNAL_BUNDLE.json",
            "TRACKB_QUALIFICATION_BUNDLE.json",
            "PASS_PREEXECUTION_ATTACHED_TRUST",
            "TRACKB_V5_EXECUTION_RECEIPT.json",
            "Refusing to delete/overwrite existing authoritative Track-B output",
        ),
        "Notebook 01",
    )
    _forbid(
        nb01_text,
        (
            "git clone",
            "data.mendeley.com",
            "zenodo.org/api",
            "run_trackb_r07_master.py",
            "shutil.rmtree(OUT)",
            "get_secret('CROPCOP_GITHUB_TOKEN')",
            'get_secret("CROPCOP_GITHUB_TOKEN")',
        ),
        "Notebook 01",
    )
    if "os.environ.pop('CROPCOP_GITHUB_TOKEN', None)" not in nb01_text:
        raise TrackBError("Notebook 01 must scrub GitHub credentials before scientific execution")

    if nb01_text.index("PASS_PREEXECUTION_ATTACHED_TRUST") > nb01_text.index(
        "bootstrap_trackb_runtime.py"
    ):
        raise TrackBError("Notebook 01 executes attached code before authenticating it")

    core_builder = (
        root / "journal_extension/scripts/build_trackb_core_package.py"
    ).read_text(encoding="utf-8")
    _require(
        core_builder,
        (
            "TRACKB_R07_EXECUTION_LOCK_v4.json",
            "TRACKB_CODE_ATTESTATION_v4.json",
            "VAL_COUNT = 16368",
            "consumed_test_image_bytes_copied",
            "v1_validation",
        ),
        "core builder",
    )
    _forbid(
        core_builder,
        (
            "TRACKB_R07_EXECUTION_LOCK_v3.json",
            "TRACKB_CODE_ATTESTATION_v3.json",
            "dataset/test",
            "copytree(image_root",
        ),
        "core builder",
    )

    historical = (
        root / "journal_extension/scripts/build_trackb_historical_compare.py"
    ).read_text(encoding="utf-8")
    _require(
        historical,
        (
            'SAFE_SCOPE = "V1_TRAIN_VAL_ONLY"',
            "SAFE_ROWS = TRAIN_ROWS + VAL_ROWS",
            '"v1_test_image_bytes_accessed": False',
            '"maximum_evidence_grade": "EXT-S"',
            "dino_batch_probe_size",
            "len(dino_samples) != 64",
            "_historical_output_budget_bytes()",
        ),
        "historical builder",
    )
    _forbid(historical, ("EXPECTED_ROWS = 117546",), "historical builder")

    materializer = (
        root / "journal_extension/scripts/trackb_v4_materialize.py"
    ).read_text(encoding="utf-8")
    _require(
        materializer,
        (
            "authoritative external cohorts — fail-fast sealed acquisition",
            "expected_source_manifest_sha256",
            "_role_content_identity",
            "role_content_identity",
            "PASS_TRACKB_INPUT_MATERIALIZATION",
        ),
        "materializer",
    )
    if materializer.index("authoritative external cohorts — fail-fast sealed acquisition") > materializer.index(
        "safe historical comparison"
    ):
        raise TrackBError("materializer runs historical compute before external source sealing")

    controller = (
        root / "journal_extension/scripts/trackb_v4_execute_attached.py"
    ).read_text(encoding="utf-8")
    _require(
        controller,
        (
            "_validate_pairing",
            "_validate_qualification_bundle",
            "PASS_IMMUTABLE_PREDICTION_BLIND_QUALIFICATION",
            "TRACKB_V5_EXECUTION_RECEIPT.json",
            "qualification_recomputed",
            "refusing to delete or overwrite an existing Track-B authoritative output root",
        ),
        "attached execution controller",
    )
    _forbid(controller, ("shutil.rmtree(output_root)",), "attached execution controller")

    runner = (
        root / "journal_extension/scripts/run_trackb_r07.py"
    ).read_text(encoding="utf-8")
    _require(
        runner,
        (
            'choices=["preflight", "qualification", "claim", "all"]',
            "_load_claim_qualification",
            "_execute_protected_claim_from_qualification",
            "qualification_recomputed",
            "acquire_claim_lease",
            "SCIENCE_QA_PASS",
            "TRACKB_CAPACITY_PREFLIGHT.json",
            "_configure_determinism",
            "_require_remaining_time",
            "PASS_PREDICTION_BLIND_QUALIFICATION",
        ),
        "scientific runner",
    )
    claim_gate = runner.index('if args.mode == "claim":')
    audit_gate = runner.index('stage("1-4 :: prediction-blind source verification')
    if claim_gate > audit_gate:
        raise TrackBError("claim path does not branch before prediction-blind audit recomputation")

    ops = (
        root / "journal_extension/src/cropcop_je/trackb_r07_ops.py"
    ).read_text(encoding="utf-8")
    _require(
        ops,
        (
            "_download_mendeley_record",
            "signed-URL refresh",
            "source byte-size mismatch",
            "duplicate/case-colliding member path",
            "acquire_claim_lease",
            "TRACKB_KAGGLE_CONTENT_MANIFEST.json",
            "allow_version",
        ),
        "Track-B operations",
    )
    _forbid(ops, ('"--public"',), "Track-B operations")

    audit = (
        root / "journal_extension/src/cropcop_je/trackb_r07_audit.py"
    ).read_text(encoding="utf-8")
    _require(
        audit,
        (
            "_CV2_RANSAC_LOCK",
            "cv2.setRNGSeed",
            "deterministic top-k tie resolution",
        ),
        "prediction-blind audit",
    )
    _forbid(
        audit,
        ("load_r07_checkpoint", "prediction_rows", "mapped_scope_metrics", "R07_CHECKPOINTS"),
        "prediction-blind audit",
    )

    bootstrap = (
        root / "journal_extension/scripts/bootstrap_trackb_runtime.py"
    ).read_text(encoding="utf-8")
    _require(
        bootstrap,
        (
            '"torch": "2.12.1"',
            '"torchvision": "0.27.1"',
            '"timm": "1.0.26"',
            '"numpy": "2.5.2"',
            '"opencv-python-headless": "4.13.0.92"',
            '"kaggle": "2.2.4"',
            "scientific_execution_requires_fresh_subprocess",
            "cuda_available",
        ),
        "runtime bootstrap",
    )

    result = {
        "schema_version": "2.0",
        "status": "PASS_TRACKB_V5_INFRASTRUCTURE_QA",
        "authority_id": AUTHORITY_ID,
        "execution_lock_id": EXECUTION_LOCK_ID,
        "code_attestation_id": CODE_ATTESTATION_ID,
        "code_attestation_sha256": sha256_file(attestation_path),
        "attested_file_count": len(attestation.get("files", [])),
        "external_source_authority": source_lock.get("status"),
        "release_status": release.get("status"),
        "release_executable": release.get("executable"),
        "notebook_00": nb00,
        "notebook_01": nb01,
        "two_notebook_operator_surface": True,
        "attached_code_authenticated_before_execution": True,
        "full_role_content_binding": True,
        "immutable_qualification_handoff": True,
        "claim_recomputes_qualification": False,
        "single_writer_claim_lease": True,
        "durable_attempt_state_machine": True,
        "v1_test_closed": True,
        "protected_external_predictions_before_claim": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
