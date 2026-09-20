from __future__ import annotations

import argparse
import json
from pathlib import Path


class ValidationError(RuntimeError):
    pass


def load_json(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValidationError(f"JSON object required: {path}")
    return obj


def notebook_text(path: Path) -> tuple[dict, str]:
    nb = load_json(path)
    if nb.get("nbformat") != 4:
        raise ValidationError(f"unexpected nbformat: {path}")
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
    )
    return nb, text


def require_all(text: str, needles: tuple[str, ...], label: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise ValidationError(f"{label} missing required safety bindings: {missing}")


def forbid_all(text: str, needles: tuple[str, ...], label: str) -> None:
    present = [needle for needle in needles if needle in text]
    if present:
        raise ValidationError(f"{label} contains forbidden/stale surfaces: {present}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-seal QA for the hardened Track-B v5 two-notebook operator surface."
    )
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()

    release = load_json(
        repo / "journal_extension/track_b_r07/TRACKB_RELEASE_AUTHORITY_v1.json"
    )
    external = load_json(
        repo / "journal_extension/track_b_r07/TRACKB_EXTERNAL_SOURCE_LOCK_v2.json"
    )
    if release.get("release_id") != "TRACKB_V5_RELEASE_AUTHORITY_v1":
        raise ValidationError("unexpected Track-B v5 release authority identity")
    if release.get("scientific_change") is not False:
        raise ValidationError("Track-B v5 remediation must preserve frozen science")
    if release.get("executable") not in {False, True}:
        raise ValidationError("Track-B release authority lacks explicit executable state")
    if external.get("lock_id") != "TRACKB_EXTERNAL_SOURCE_LOCK_v2":
        raise ValidationError("unexpected external-source authority")
    if external.get("status") != "PASS_EXTERNAL_SOURCE_AUTHORITY":
        raise ValidationError("external-source authority is not PASS")
    blocked = [
        role
        for role, row in (external.get("candidates") or {}).items()
        if not isinstance(row, dict) or row.get("materialization_permitted") is not True
    ]
    if blocked:
        raise ValidationError(f"external candidates remain blocked: {blocked}")

    readiness_path = (
        repo / "journal_extension/kaggle/TrackB_00_Readiness_Materialization.ipynb"
    )
    final_path = repo / "journal_extension/kaggle/TrackB_01_Final_Execution.ipynb"
    readiness, readiness_text = notebook_text(readiness_path)
    final, final_text = notebook_text(final_path)

    orchestration = load_json(
        repo / "journal_extension/track_b_r07/TRACKB_ORCHESTRATION_LOCK_v5.json"
    )
    runtime_source_commit = str(orchestration.get("runtime_source_commit", "")).strip()
    if len(runtime_source_commit) != 40:
        raise ValidationError("v5 orchestration lock lacks exact runtime source commit")
    if f"SOURCE_COMMIT = '{runtime_source_commit}'" not in readiness_text:
        raise ValidationError(
            "Notebook 00 runtime pin differs from TRACKB_ORCHESTRATION_LOCK_v5"
        )
    if f"RUNTIME_SOURCE_COMMIT = '{runtime_source_commit}'" not in final_text:
        raise ValidationError(
            "Notebook 01 runtime pin differs from TRACKB_ORCHESTRATION_LOCK_v5"
        )
    if "RELEASE_ID = 'TRACKB_V5_RELEASE_AUTHORITY_v1'" not in final_text:
        raise ValidationError("Notebook 01 does not bind the v5 release authority identity")

    if len(readiness.get("cells", [])) != 5:
        raise ValidationError("readiness notebook must have exactly five cells")
    if len(final.get("cells", [])) != 6:
        raise ValidationError("final execution notebook must have exactly six cells")

    require_all(
        readiness_text,
        (
            "trackb_v4_materialize.py",
            "KAGGLE_API_TOKEN",
            "source_qualification_sha256",
            "protected_external_prediction_count",
            "v1_test_accessed",
        ),
        "Notebook 00",
    )
    forbid_all(
        readiness_text,
        (
            "CROPCOP_GITHUB_TOKEN",
            "run_trackb_r07_master.py",
        ),
        "Notebook 00",
    )

    require_all(
        final_text,
        (
            "RUN_MODE = 'qualification'",
            "TRACKB_INFRASTRUCTURE_BUNDLE.json",
            "TRACKB_EXTERNAL_BUNDLE.json",
            "TRACKB_QUALIFICATION_BUNDLE.json",
            "PASS_PREEXECUTION_ATTACHED_TRUST",
            "_role_identity",
            "_git_blob_sha1",
            "requirements-trackb.lock.txt",
            "trackb_v4_execute_attached.py",
            "TRACKB_V5_EXECUTION_RECEIPT.json",
            "qualification_recomputed",
            "Refusing to delete/overwrite existing authoritative Track-B output",
        ),
        "Notebook 01",
    )
    forbid_all(
        final_text,
        (
            "data.mendeley.com",
            "zenodo.org/api",
            "kaggle datasets download",
            "git clone",
            "run_trackb_r07_master.py",
            "shutil.rmtree(OUT)",
            "get_secret('CROPCOP_GITHUB_TOKEN')",
            'get_secret("CROPCOP_GITHUB_TOKEN")',
        ),
        "Notebook 01",
    )

    preexec = final_text.index("PASS_PREEXECUTION_ATTACHED_TRUST")
    bootstrap = final_text.index("BOOTSTRAP = REPO / 'journal_extension/scripts/bootstrap_trackb_runtime.py'")
    if preexec > bootstrap:
        raise ValidationError(
            "Notebook 01 must authenticate attached bytes/code before runtime bootstrap"
        )

    pop_token = final_text.index("os.environ.pop('KAGGLE_API_TOKEN', None)")
    controller = final_text.index(
        "CONTROLLER = REPO / 'journal_extension/scripts/trackb_v4_execute_attached.py'"
    )
    if pop_token > controller:
        raise ValidationError(
            "Notebook 01 must clear inherited Kaggle credentials before scientific controller launch"
        )

    materializer_source = (
        repo / "journal_extension/scripts/trackb_v4_materialize.py"
    ).read_text(encoding="utf-8")
    controller_source = (
        repo / "journal_extension/scripts/trackb_v4_execute_attached.py"
    ).read_text(encoding="utf-8")
    runner_source = (
        repo / "journal_extension/scripts/run_trackb_r07.py"
    ).read_text(encoding="utf-8")
    ops_source = (
        repo / "journal_extension/src/cropcop_je/trackb_r07_ops.py"
    ).read_text(encoding="utf-8")
    audit_source = (
        repo / "journal_extension/src/cropcop_je/trackb_r07_audit.py"
    ).read_text(encoding="utf-8")

    require_all(
        materializer_source,
        (
            "authoritative external cohorts — fail-fast sealed acquisition",
            "expected_source_manifest_sha256",
            "role_content_identity",
            "_role_content_identity",
        ),
        "materializer",
    )
    require_all(
        controller_source,
        (
            "PASS_IMMUTABLE_PREDICTION_BLIND_QUALIFICATION",
            "_validate_qualification_bundle",
            "**result",
            "TRACKB_V5_EXECUTION_RECEIPT.json",
            "refusing to delete or overwrite an existing Track-B authoritative output root",
        ),
        "attached-input controller",
    )
    require_all(
        runner_source,
        (
            "_load_claim_qualification",
            "_execute_protected_claim_from_qualification",
            "qualification_recomputed",
            "acquire_claim_lease",
            "TRACKB_CAPACITY_PREFLIGHT.json",
            "_configure_determinism",
            "_require_remaining_time",
        ),
        "scientific runner",
    )
    require_all(
        ops_source,
        (
            "_download_mendeley_record",
            "signed-URL refresh",
            "source byte-size mismatch",
            "duplicate/case-colliding member path",
            "acquire_claim_lease",
            "allow_version",
        ),
        "Track-B operations",
    )
    require_all(
        audit_source,
        (
            "_CV2_RANSAC_LOCK",
            "cv2.setRNGSeed",
            "deterministic top-k tie resolution",
        ),
        "Track-B audit implementation",
    )

    result = {
        "status": "PASS_TRACKB_V5_PRESEAL_ORCHESTRATION_QA",
        "scientific_change": False,
        "release_status": release.get("status"),
        "release_executable": release.get("executable"),
        "external_source_status": external.get("status"),
        "runtime_source_commit": runtime_source_commit,
        "operator_runtime_binding": "PASS",
        "notebook_01_trust_before_execute": True,
        "qualification_handoff_required_for_claim": True,
        "claim_recomputes_qualification": False,
        "protected_external_predictions_before_claim": 0,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
