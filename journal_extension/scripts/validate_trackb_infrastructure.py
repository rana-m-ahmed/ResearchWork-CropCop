from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import (
    AUTHORITY_ID,
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
    return {"cell_count": len(nb.get("cells", [])), "code_cell_count": code_cells, "text": "\n".join(combined)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Static QA gate for the lean CropCop Track-B infrastructure.")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    authority_path = root / "journal_extension/amendments/track_bc_r07_downstream_v2.json"
    lock_path = root / "journal_extension/track_b_r07/TRACKB_R07_EXECUTION_LOCK_v2.json"
    attestation_path = root / "journal_extension/track_b_r07/TRACKB_CODE_ATTESTATION_v2.json"
    final_nb = root / "journal_extension/kaggle/trackb_r07_end_to_end.ipynb"
    master_nb = root / "journal_extension/kaggle/trackb_r07_master.ipynb"
    core_nb = root / "journal_extension/kaggle/trackb_build_core_package.ipynb"
    hist_nb = root / "journal_extension/kaggle/trackb_build_historical_compare.ipynb"
    runner = root / "journal_extension/scripts/run_trackb_r07.py"
    audit_module = root / "journal_extension/src/cropcop_je/trackb_r07_audit.py"
    ops_module = root / "journal_extension/src/cropcop_je/trackb_r07_ops.py"
    master_runner = root / "journal_extension/scripts/run_trackb_r07_master.py"
    runtime_bootstrap = root / "journal_extension/scripts/bootstrap_trackb_runtime.py"
    core_builder = root / "journal_extension/scripts/build_trackb_core_package.py"
    hist_builder = root / "journal_extension/scripts/build_trackb_historical_compare.py"
    hist_source_prep = root / "journal_extension/scripts/prepare_trackb_historical_source_input.py"

    for path in (
        authority_path, lock_path, attestation_path, final_nb, master_nb, core_nb, hist_nb, runner,
        audit_module, ops_module, master_runner, runtime_bootstrap, core_builder, hist_builder, hist_source_prep,
    ):
        if not path.is_file():
            raise TrackBError(f"required Track-B infrastructure file missing: {path.relative_to(root)}")

    authority = load_json(authority_path)
    lock = load_json(lock_path)
    if authority.get("authority_id") != AUTHORITY_ID or lock.get("lock_id") != EXECUTION_LOCK_ID:
        raise TrackBError("Track-B authority/lock identity mismatch")
    validate_downstream_authority(authority)
    validate_execution_lock(lock)
    if sha256_file(attestation_path) != lock.get("code_attestation_sha256"):
        raise TrackBError("execution lock does not bind the committed code attestation")
    attestation = verify_code_attestation(root, attestation_path)

    final_info = _compile_notebook(final_nb)
    master_info = _compile_notebook(master_nb)
    core_info = _compile_notebook(core_nb)
    hist_info = _compile_notebook(hist_nb)
    final_text = final_info.pop("text")
    master_text = master_info.pop("text")
    core_text = core_info.pop("text")
    hist_text = hist_info.pop("text")
    lowered = final_text.lower()
    for forbidden in ("git push", "github_token", "gh_token", "ds-v1-test-consumed", "--split test"):
        if forbidden in lowered:
            raise TrackBError(f"final Track-B notebook contains forbidden surface/action: {forbidden}")
    for required in ("run_trackb_r07.py", "--mode", "all", "TRACKB_FINAL_QA.json", "TRACK_B_CLOSED"):
        if required not in final_text:
            raise TrackBError(f"final Track-B notebook missing expected controller binding: {required}")
    for required in (
        "build_trackb_core_package.py",
        "16368",
        "sec-je-r07-cnxtt-context-s1-8904b100d223-a01",
        "cropcop-r07-cnxtt-context-s2-abce1197-56023042",
        "cropcop-r07-cnxtt-context-s3-f13ca687-56023042",
        "cropcop-secondary-g1-8904b100",
    ):
        if required not in core_text:
            raise TrackBError(f"core package notebook missing frozen operator binding: {required}")
    if "run_trackb_r07.py" in core_text or "_protected_inference" in core_text:
        raise TrackBError("core package notebook exposes protected classifier execution")

    core_builder_text = core_builder.read_text(encoding="utf-8")
    for required in (
        "VAL_COUNT = 16368",
        "R07_RUN_RECORDS",
        "DINO_FACTORY_MANIFEST_SHA256",
        "consumed_test_image_bytes_copied",
        "v1_validation",
    ):
        if required not in core_builder_text:
            raise TrackBError(f"core package builder missing safety/identity guard: {required}")
    if "copytree(image_root" in core_builder_text or "dataset/test" in core_builder_text:
        raise TrackBError("core builder contains a broad or explicit consumed-test copy path")

    for required in (
        "KAGGLE_API_TOKEN",
        "CROPCOP_GITHUB_TOKEN",
        "run_trackb_r07_master.py",
        "bootstrap_trackb_runtime.py",
        "requirements-trackb.lock.txt",
        "PASS_AUTOMATED_TRACK_B_COMPLETE",
    ):
        if required not in master_text:
            raise TrackBError(f"master notebook missing automation binding: {required}")
    if "agrivision_v2" in master_text.lower() or "agrivision_bd" in master_text.lower():
        raise TrackBError("master notebook still references retired Agri-Vision candidate")
    if "metadata.version(" in master_text:
        raise TrackBError("master notebook still depends on Kaggle live-kernel package versions")
    if "sys.executable" not in master_text or "--requirements" not in master_text:
        raise TrackBError("master notebook does not use the proven active-interpreter lock-repair path")
    if "VENV_PY" in master_text or "trackb_runtime_env" in master_text:
        raise TrackBError("master notebook still exposes the failed venv execution path")

    ops_text = ops_module.read_text(encoding="utf-8")
    for required in (
        "KAGGLE_API_TOKEN",
        "CROPCOP_GITHUB_TOKEN",
        "publish_to_github_branch",
        "publish_private_kaggle_dataset",
        "acquire_gvlid_v5",
        "acquire_irish_potato",
        "detect_authenticated_kaggle_owner",
        "verify_authenticated_kaggle_owner",
        "verify_kaggle_source_access",
        "verify_github_repository_push_access",
        "probe_external_sources",
        "normalized_image_count",
        "observed_class_support",
        "raw_external_images_included",
    ):
        if required not in ops_text:
            raise TrackBError(f"Track-B operations module missing automation/security guard: {required}")
    if "--public" in ops_text or '"-u"' in ops_text:
        raise TrackBError("Track-B operations module exposes public Kaggle dataset publication")
    if '_safe_extract_zip(archive, data_root / class_name)' in ops_text:
        raise TrackBError("Irish Potato acquisition still trusts archive-internal directory layout")

    bootstrap_text = runtime_bootstrap.read_text(encoding="utf-8")
    for required in (
        '"torch": "2.12.1"',
        '"torchvision": "0.27.1"',
        '"timm": "1.0.26"',
        '"numpy": "2.5.2"',
        '"opencv-python-headless": "4.13.0.92"',
        '"kaggle": "2.2.4"',
        "requirements-trackb.lock.txt",
        '"-m",',
        '"pip",',
        '"install",',
        '"--no-cache-dir"',
        "scientific_execution_requires_fresh_subprocess",
        "cuda_available",
        "opencv_runtime_version",
        "pre_install_conflicting_opencv_variants",
        "--force-reinstall",
        "--no-deps",
    ):
        if required not in bootstrap_text:
            raise TrackBError(f"Track-B runtime repair bootstrap missing frozen guard: {required}")
    for forbidden in ("-m\", \"venv", "-m\", \"ensurepip", "VENV_PY", "trackb_runtime_env"):
        if forbidden in bootstrap_text:
            raise TrackBError(f"Track-B runtime bootstrap still exposes failed venv path: {forbidden}")

    master_runner_text = master_runner.read_text(encoding="utf-8")
    for required in (
        "configure_runtime_secrets",
        "download_kaggle_dataset",
        "verify_authenticated_kaggle_owner",
        "verify_kaggle_source_access",
        "verify_github_repository_push_access",
        "probe_external_sources",
        "historical_dataset_slug",
        "evidence_dataset_slug",
        "publish_public_trackb_evidence",
        "publish_private_kaggle_dataset",
        "stage(\"2.5 :: release redundant model-source downloads\")",
        "source_roots[\"final_v1\"]",
        "gvlid_v5",
        "irish_potato",
    ):
        if required not in master_runner_text:
            raise TrackBError(f"master controller missing automated operation: {required}")
    if "agrivision_v2" in master_runner_text.lower() or "agrivision_bd" in master_runner_text.lower():
        raise TrackBError("master controller still references retired Agri-Vision candidate")
    if '"--kaggle-owner", default=KAGGLE_OWNER_DEFAULT' not in master_runner_text:
        raise TrackBError("master controller does not expose the authenticated-owner policy surface")
    automation = lock.get("automation", {})
    if automation.get("kaggle_dataset_owner_default") != "AUTO":
        raise TrackBError("execution lock does not default private Kaggle ownership to AUTO")
    if automation.get("kaggle_dataset_owner_mode") != "AUTHENTICATED_TOKEN_OWNER_AUTO_DETECT":
        raise TrackBError("execution lock does not require authenticated Kaggle owner auto-detection")
    kaggle_policy = lock.get("kaggle", {})
    if kaggle_policy.get("internet_required_during_claim_run") is not True:
        raise TrackBError("automated master requires Internet for orchestration/publication")
    if kaggle_policy.get("protected_inference_network_dependency") is not False:
        raise TrackBError("protected inference must not depend scientifically on network access")

    if "build_trackb_historical_compare.py" not in hist_text or "code_attestation" not in hist_text:
        raise TrackBError("historical comparison notebook is not bound to the attested builder")
    for required in (
        "V1_TRAIN_VAL_ONLY",
        "92744",
        "EXT-S",
        "v1_test_image_bytes_accessed",
        "--v1-manifest",
    ):
        if required not in hist_text:
            raise TrackBError(f"safe historical comparison notebook missing guard: {required}")
    if "--source-manifest" in hist_text or "historical_source package" in hist_text:
        raise TrackBError("historical comparison notebook still exposes the obsolete full-raw source route")

    audit_text = audit_module.read_text(encoding="utf-8")
    for forbidden in ("load_r07_checkpoint", "prediction_rows", "mapped_scope_metrics", "R07_CHECKPOINTS"):
        if forbidden in audit_text:
            raise TrackBError(f"prediction-blind audit module imports/references classifier path: {forbidden}")

    hist_builder_text = hist_builder.read_text(encoding="utf-8")
    for required in (
        'SAFE_ROWS = TRAIN_ROWS + VAL_ROWS',
        'SAFE_SCOPE = "V1_TRAIN_VAL_ONLY"',
        '"v1_test_image_bytes_accessed": False',
        '"maximum_evidence_grade": "EXT-S"',
    ):
        if required not in hist_builder_text:
            raise TrackBError(f"safe historical builder missing guard: {required}")
    if "EXPECTED_ROWS = 117546" in hist_builder_text:
        raise TrackBError("post-closure historical builder still authorizes raw 117,546-image reconstruction")
    prep_text = hist_source_prep.read_text(encoding="utf-8")
    if "Raw 117,546-image historical-source reconstruction is disabled after V1-test closure" not in prep_text:
        raise TrackBError("obsolete full-raw historical source preparer is not fail-closed")

    runner_text = runner.read_text(encoding="utf-8")
    for required in (
        "declared_hist_count == 92744",
        '"coverage_scope": "V1_TRAIN_VAL_ONLY"',
        '"maximum_evidence_grade": "EXT-S"',
        "full 117,546-image EXT-I route is dormant",
        "locked Python drift",
    ):
        if required not in runner_text:
            raise TrackBError(f"final runner missing safe historical-package gate: {required}")
    audit_call = runner_text.find('grape = _audit_candidate(')
    second_audit_call = runner_text.find('potato = _audit_candidate(')
    firewall = runner_text.find('stage("4 :: prediction firewall")')
    protected = runner_text.find('"gvlid_grape": _protected_inference(')
    if min(audit_call, second_audit_call, firewall, protected) < 0 or not (audit_call < second_audit_call < firewall < protected):
        raise TrackBError("runner chronology no longer guarantees both v2 audits before protected inference")
    if "agrivision" in runner_text.lower():
        raise TrackBError("executable runner still contains a retired Agri-Vision code path/token")
    if "Tomato Mosaic -> tomato_mosaic_virus".lower() in runner_text.lower():
        raise TrackBError("executable runner still contains the retired Tomato Mosaic semantic gate")

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "authority_id": AUTHORITY_ID,
        "execution_lock_id": EXECUTION_LOCK_ID,
        "code_attestation_sha256": sha256_file(attestation_path),
        "attested_file_count": len(attestation.get("files", [])),
        "final_notebook": final_info,
        "master_notebook": master_info,
        "core_builder_notebook": core_info,
        "historical_builder_notebook": hist_info,
        "single_master_automation_surface": True,
        "proven_kaggle_lock_repair_bootstrap": True,
        "automatic_private_kaggle_archival": True,
        "automatic_public_safe_github_publication": True,
        "track_b_v2_candidates": ["gvlid_grape", "irish_potato"],
        "retired_v1_candidate_absent": True,
        "core_builder_validation_only_surface": True,
        "prediction_blind_audit_module": True,
        "both_candidate_audits_before_protected_inference": True,
        "git_push_from_claim_notebook": False,
        "consumed_v1_test_reference_in_claim_notebook": False,
        "safe_historical_builder_scope": "V1_TRAIN_VAL_ONLY",
        "safe_historical_builder_image_count": 92744,
        "safe_historical_builder_maximum_grade": "EXT-S",
        "full_ext_i_representation_policy": "pre_test_recovered_representation_only",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
