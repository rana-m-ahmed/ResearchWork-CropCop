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

    authority_path = root / "journal_extension/amendments/track_bc_r07_downstream_v3.json"
    lock_path = root / "journal_extension/track_b_r07/TRACKB_R07_EXECUTION_LOCK_v3.json"
    attestation_path = root / "journal_extension/track_b_r07/TRACKB_CODE_ATTESTATION_v3.json"
    final_nb = root / "journal_extension/kaggle/trackb_r07_end_to_end.ipynb"
    master_nb = root / "journal_extension/kaggle/trackb_r07_master.ipynb"
    core_nb = root / "journal_extension/kaggle/trackb_build_core_package.ipynb"
    hist_nb = root / "journal_extension/kaggle/trackb_build_historical_compare.ipynb"
    runner = root / "journal_extension/scripts/run_trackb_r07.py"
    audit_module = root / "journal_extension/src/cropcop_je/trackb_r07_audit.py"
    ops_module = root / "journal_extension/src/cropcop_je/trackb_r07_ops.py"
    master_runner = root / "journal_extension/scripts/run_trackb_r07_master.py"
    preinference_validator = root / "journal_extension/scripts/validate_trackb_preinference_qualification.py"
    runtime_bootstrap = root / "journal_extension/scripts/bootstrap_trackb_runtime.py"
    core_builder = root / "journal_extension/scripts/build_trackb_core_package.py"
    hist_builder = root / "journal_extension/scripts/build_trackb_historical_compare.py"
    hist_source_prep = root / "journal_extension/scripts/prepare_trackb_historical_source_input.py"

    for path in (
        authority_path, lock_path, attestation_path, final_nb, master_nb, core_nb, hist_nb, runner,
        audit_module, ops_module, master_runner, preinference_validator, runtime_bootstrap, core_builder, hist_builder, hist_source_prep,
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
    for required in ("run_trackb_r07.py", "--mode", "preflight", "PREFLIGHT_PASS.json"):
        if required not in final_text:
            raise TrackBError(f"internal Track-B notebook missing prediction-free preflight binding: {required}")
    for forbidden in ("--mode', 'all", "--mode\", \"all", "PROTECTED_INFERENCE_STARTED", "CROPCOP_GITHUB_TOKEN"):
        if forbidden in final_text:
            raise TrackBError(f"internal Track-B notebook exposes forbidden protected-run surface: {forbidden}")
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
        "run_trackb_r07_master.py",
        "bootstrap_trackb_runtime.py",
        "requirements-trackb.lock.txt",
        "PASS_TRACKB_PREINFERENCE_QUALIFICATION",
        "TRACKB_PREINFERENCE_QUALIFICATION.json",
        "qualification_science_sha256",
        "TRACKB_PREINFERENCE_QA.json",
        "--execution-mode",
        "qualification",
        "/kaggle/tmp/cropcop_trackb_r07",
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
    if "CROPCOP_GITHUB_TOKEN" in master_text or "verify_github_repository_push_access" in master_text:
        raise TrackBError("qualification notebook must not require GitHub credentials or push preflight")
    if "SKIPPED for prediction-blind qualification" not in master_text:
        raise TrackBError("qualification notebook does not explicitly record skipped GitHub preflight")
    if "'--execution-mode', 'claim'" in master_text or '"--execution-mode", "claim"' in master_text:
        raise TrackBError("operator notebook exposes protected claim mode instead of qualification mode")
    if "--authorized-qualification-science-sha256" in master_text:
        raise TrackBError("qualification notebook exposes protected-claim authorization input")
    frozen_qualification_sha = "73e07bb23d94e67bd2ad8f9d63226cd4488caf70"
    if f"SOURCE_COMMIT = '{frozen_qualification_sha}'" not in master_text:
        raise TrackBError("qualification notebook is not pinned to the audited implementation SHA")
    if "SOURCE_REF" in master_text or "checkout', '--detach'" not in master_text:
        raise TrackBError("qualification notebook regressed to mutable branch execution")

    ops_text = ops_module.read_text(encoding="utf-8")
    for required in (
        "KAGGLE_API_TOKEN",
        "CROPCOP_GITHUB_TOKEN",
        "publish_to_github_branch",
        "publish_private_kaggle_dataset",
        "acquire_gvlid_v5",
        "MENDELEY_PUBLIC_API",
        "_mendeley_public_file_records",
        "acquire_irish_potato",
        "detect_authenticated_kaggle_owner",
        "verify_authenticated_kaggle_owner",
        "verify_kaggle_source_access",
        "verify_github_repository_push_access",
        "probe_external_sources",
        "normalized_image_count",
        "observed_class_support",
        "raw_external_images_included",
        "TRACKB_KAGGLE_CONTENT_MANIFEST.json",
        "source_checksum_verified",
        "TRACKB_EXTERNAL_LINEAGE_REVIEW_v1",
        "attempt_dataset_slug",
        "publish_attempt_state",
        "read_latest_attempt_state",
    ):
        if required not in ops_text:
            raise TrackBError(f"Track-B operations module missing automation/security guard: {required}")
    if "--public" in ops_text or '"-u"' in ops_text:
        raise TrackBError("Track-B operations module exposes public Kaggle dataset publication")
    for required in (
        '"push", "--dry-run"',
        "GIT_PUSH_DRY_RUN_NO_REMOTE_REF_CREATED",
        "GIT_ASKPASS_REQUIRE",
        "credential.helper",
        "_classify_github_push_failure",
    ):
        if required not in ops_text:
            raise TrackBError(f"GitHub write preflight missing exact Git-transport guard: {required}")
    if "api.github.com/repos/" in ops_text:
        raise TrackBError("Track-B GitHub write preflight regressed to REST-only permission probing")
    if '_safe_extract_zip(archive, data_root / class_name)' in ops_text:
        raise TrackBError("Irish Potato acquisition still trusts archive-internal directory layout")
    for required in (
        "Private Kaggle publication attempt",
        "for attempt, delay in enumerate((0, 5, 15, 30), start=1)",
        "private Kaggle dataset publication failed after 4 attempts",
        "reused_identical_remote",
        "_verify_remote_kaggle_content",
        "_wait_kaggle_dataset_ready",
    ):
        if required not in ops_text:
            raise TrackBError(f"private Kaggle publication retry guard missing: {required}")

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
        "/kaggle/tmp/cropcop_trackb_r07",
        "attempt_dataset_slug",
        "PRIVATE_ARCHIVE_VERIFIED",
        "PUBLICATION_COMPLETE",
        "full_roundtrip=True",
        "validate_trackb_preinference_qualification.py",
        "PASS_TRACKB_PREINFERENCE_QUALIFICATION",
        "--authorized-qualification-science-sha256",
        '"--execution-mode", choices=["qualification", "claim"], default="qualification"',
        "configure_runtime_secrets(require_github=claim_mode)",
        "SKIPPED_QUALIFICATION_MODE",
    ):
        if required not in master_runner_text:
            raise TrackBError(f"master controller missing automated operation: {required}")
    if "agrivision_v2" in master_runner_text.lower() or "agrivision_bd" in master_runner_text.lower():
        raise TrackBError("master controller still references retired Agri-Vision candidate")
    archive_pos = master_runner_text.find('stage("7 :: restricted evidence archive to private Kaggle")')
    github_pos = master_runner_text.find('stage("8 :: safe GitHub evidence publication")')
    if min(archive_pos, github_pos) < 0 or archive_pos > github_pos:
        raise TrackBError("private evidence must be durably archived before GitHub publication")
    for required in (
        "TRACK_B_CLOSED_PRIVATE_EVIDENCE_ARCHIVED_GITHUB_PUBLICATION_FAILED",
        "for attempt, delay in enumerate((0, 5, 15, 30), start=1)",
        "scientific_closure_durable_before_github_publication",
    ):
        if required not in master_runner_text:
            raise TrackBError(f"master controller missing publication durability guard: {required}")
    if '"--kaggle-owner", default=KAGGLE_OWNER_DEFAULT' not in master_runner_text:
        raise TrackBError("master controller does not expose the authenticated-owner policy surface")
    automation = lock.get("automation", {})
    if automation.get("kaggle_dataset_owner_default") != "AUTO":
        raise TrackBError("execution lock does not default private Kaggle ownership to AUTO")
    if automation.get("kaggle_dataset_owner_mode") != "AUTHENTICATED_TOKEN_OWNER_AUTO_DETECT":
        raise TrackBError("execution lock does not require authenticated Kaggle owner auto-detection")
    if automation.get("github_token_required_modes") != ["claim"]:
        raise TrackBError("GitHub token must be claim-only")
    if automation.get("github_push_preflight_modes") != ["claim"]:
        raise TrackBError("GitHub push preflight must be claim-only")
    if automation.get("qualification_requires_github_token") is not False:
        raise TrackBError("qualification must not require GitHub token")
    if automation.get("qualification_requires_github_push_preflight") is not False:
        raise TrackBError("qualification must not require GitHub push preflight")
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
        "audit_policy_from_lock",
        "deterministic_representative_order",
        "PROTECTED_INFERENCE_STARTED",
        "TRACKB_SCIENCE_MANIFEST.json",
        "trackb_science_sha256",
        "_verify_zip_archive",
        "PASS_PREDICTION_BLIND_QUALIFICATION",
        "TRACKB_PREINFERENCE_QUALIFICATION.json",
        'choices=["preflight", "qualification", "all"]',
    ):
        if required not in runner_text:
            raise TrackBError(f"final runner missing safe historical-package gate: {required}")
    package_pos = runner_text.find('packages = _package_outputs(output_root)')
    closure_write_pos = runner_text.find('atomic_write_json(output_root / "TRACKB_FINAL_CLOSURE.json"')
    if min(package_pos, closure_write_pos) < 0 or package_pos > closure_write_pos:
        raise TrackBError("TRACK_B_CLOSED can be written before evidence-package verification")
    if "external_predictions_before_firewall" in runner_text:
        raise TrackBError("runner still exposes ambiguous cross-attempt prediction chronology field")
    if "def _science_preimage_manifest(" in runner_text:
        raise TrackBError("runner regressed to obsolete/timestamp-unsafe science preimage helper")
    prediction_blind_start = runner_text.find("def _prediction_blind_science_manifest(")
    stable_manifest_start = runner_text.find("def _stable_science_manifest(")
    prediction_blind_text = runner_text[prediction_blind_start:stable_manifest_start]
    for forbidden in ("sealed_at_utc", "retrieved_at", "final_qa_sha256", "candidate_input_manifest_sha256", "source_metadata_record_sha256"):
        if forbidden in prediction_blind_text:
            raise TrackBError(
                f"prediction-blind science identity contains execution/transport metadata: {forbidden}"
            )
    if "cache: dict[str, dict[str, Any]] = {}" in runner_text:
        raise TrackBError("runner regressed to an all-in-RAM candidate ORB cache")
    if "open_memmap" not in runner_text or "mmap_mode=\"r\"" not in runner_text:
        raise TrackBError("runner lacks file-backed packed candidate ORB cache")

    qualification_gate = runner_text.find('if args.mode == "qualification":')
    claim_gate = runner_text.find("claim_candidates =")
    protected_call = runner_text.find('"gvlid_grape": _protected_inference(')
    if min(qualification_gate, claim_gate, protected_call) < 0 or not (
        qualification_gate < claim_gate < protected_call
    ):
        raise TrackBError("prediction-blind qualification does not return before protected inference")

    preinference_text = preinference_validator.read_text(encoding="utf-8")
    for required in (
        "PASS_INDEPENDENT_PREINFERENCE_QA",
        "TRACKB_PREDICTION_BLIND_SCIENCE.json",
        "qualification_science_sha256",
        "reconstructed_science",
        "protected_external_prediction_count",
        "TRACKB_FINAL_CLOSURE.json",
        "TRACKB_ATTEMPT_STATE.json",
        "verify_candidate_seal",
        "V1_TRAIN_VAL_ONLY",
        "92744",
    ):
        if required not in preinference_text:
            raise TrackBError(f"pre-inference validator missing safety check: {required}")

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
        "git_write_preflight_uses_publication_transport": True,
        "git_write_preflight_before_runtime_repair": True,
        "private_evidence_archived_before_github_publication": True,
        "github_publication_retry_attempts": 4,
        "private_kaggle_publication_retry_attempts": 4,
        "track_b_v3_candidates": ["gvlid_grape", "irish_potato"],
        "lock_is_executable_policy_source": True,
        "seeded_family_order_is_operational": True,
        "source_checksums_required": True,
        "scratch_storage_isolation_required": True,
        "durable_attempt_ledger_required": True,
        "closure_after_package_verification": True,
        "stable_science_digest_required": True,
        "private_kaggle_roundtrip_required": True,
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
        "operator_default_mode": "qualification",
        "independent_preinference_validator": True,
        "protected_inference_fail_closed_until_qualification_review": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
