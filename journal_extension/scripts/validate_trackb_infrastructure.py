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

    authority_path = root / "journal_extension/amendments/track_bc_r07_downstream_v1.json"
    lock_path = root / "journal_extension/track_b_r07/TRACKB_R07_EXECUTION_LOCK_v1.json"
    attestation_path = root / "journal_extension/track_b_r07/TRACKB_CODE_ATTESTATION_v1.json"
    final_nb = root / "journal_extension/kaggle/trackb_r07_end_to_end.ipynb"
    hist_nb = root / "journal_extension/kaggle/trackb_build_historical_compare.ipynb"
    runner = root / "journal_extension/scripts/run_trackb_r07.py"
    audit_module = root / "journal_extension/src/cropcop_je/trackb_r07_audit.py"
    hist_builder = root / "journal_extension/scripts/build_trackb_historical_compare.py"
    hist_source_prep = root / "journal_extension/scripts/prepare_trackb_historical_source_input.py"

    for path in (
        authority_path, lock_path, attestation_path, final_nb, hist_nb, runner,
        audit_module, hist_builder, hist_source_prep,
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
    hist_info = _compile_notebook(hist_nb)
    final_text = final_info.pop("text")
    hist_text = hist_info.pop("text")
    lowered = final_text.lower()
    for forbidden in ("git push", "github_token", "gh_token", "ds-v1-test-consumed", "--split test"):
        if forbidden in lowered:
            raise TrackBError(f"final Track-B notebook contains forbidden surface/action: {forbidden}")
    for required in ("run_trackb_r07.py", "--mode", "all", "TRACKB_FINAL_QA.json", "TRACK_B_CLOSED"):
        if required not in final_text:
            raise TrackBError(f"final Track-B notebook missing expected controller binding: {required}")
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
    audit_call = runner_text.find('potato = _audit_candidate(')
    second_audit_call = runner_text.find('agrivision = _audit_candidate(')
    firewall = runner_text.find('stage("4 :: prediction firewall")')
    protected = runner_text.find('"irish_potato": _protected_inference(')
    if min(audit_call, second_audit_call, firewall, protected) < 0 or not (audit_call < second_audit_call < firewall < protected):
        raise TrackBError("runner chronology no longer guarantees both audits before protected inference")

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "authority_id": AUTHORITY_ID,
        "execution_lock_id": EXECUTION_LOCK_ID,
        "code_attestation_sha256": sha256_file(attestation_path),
        "attested_file_count": len(attestation.get("files", [])),
        "final_notebook": final_info,
        "historical_builder_notebook": hist_info,
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
