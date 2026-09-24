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

    for label, notebook in (("Notebook 00", readiness), ("Notebook 01", final)):
        for index, cell in enumerate(notebook.get("cells") or []):
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source") or [])
            try:
                compile(source, f"{label} cell {index}", "exec")
            except SyntaxError as exc:
                raise ValidationError(
                    f"{label} code cell {index} does not compile: {exc}"
                ) from exc

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
    if f"EXPECTED_RUNTIME_SOURCE_COMMIT = '{runtime_source_commit}'" not in final_text:
        raise ValidationError(
            "Notebook 01 runtime pin differs from TRACKB_ORCHESTRATION_LOCK_v5"
        )
    import re
    bare_runtime_symbol = re.search(r"(?<!EXPECTED_)\\bRUNTIME_SOURCE_COMMIT\\b", final_text)
    if bare_runtime_symbol:
        raise ValidationError(
            "Notebook 01 contains undefined/legacy bare RUNTIME_SOURCE_COMMIT reference"
        )
    if "EXPECTED_EXPECTED_RUNTIME_SOURCE_COMMIT" in final_text:
        raise ValidationError(
            "Notebook 01 contains malformed doubled runtime-source symbol"
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
            "PASS_PRECONTROLLER_ATTACHED_AUTHORITY",
            "_role_identity",
            "_git_blob_sha1",
            "requirements-trackb.lock.txt",
            "trackb_v4_execute_attached.py",
            "TRACKB_V5_EXECUTION_RECEIPT.json",
            "qualification_recomputed",
            "Refusing to delete/overwrite existing authoritative Track-B output",
            "KAGGLE_OWNER = 'ranamuhammadahmed6'",
            "PASS_OPERATOR_PREFLIGHT",
            "verify_kaggle_publication_capability",
            "verify_authenticated_kaggle_owner",
            "BOUND_MANIFEST_AND_EXACT_BYTE_ROUNDTRIP",
            "roundtrip_verified_file_count",
            "Track-B v5 qualification requires Kaggle T4 x2",
            "Locked Track-B runtime does not expose T4 x2",
            "Qualification mode must attach only infrastructure + external handoffs",
            "Infrastructure and external handoffs must be distinct Kaggle datasets",
            "is outside its paired dataset root",
            "DEFERRED_TO_CONTROLLER_BEFORE_SCIENCE",
            "PASS_FAST_INPUT_DISCOVERY_SMOKE",
            "TRACKB_V5_V1_GUARD_FALSE_POSITIVE_FIX_v1",
            "trackb_v5_v1_guard_hotfix.py",
            "sitecustomize.py",
            "TRACKB_V1_GUARD_HOTFIX_ACTIVE",
            "EXACT_EQUIVALENT_VECTORIZED_CUTOFF_TIE_DETECTION",
            "science_subprocess_live_streaming",
            "DINO top-k",
        ),
        "Notebook 01",
    )
    if "observed_content = {role: _role_identity(path.parent)" in final_text:
        raise ValidationError(
            "Notebook 01 redundantly rehashes all attached role bytes before controller; "
            "the controller must be the single full-byte verification gate before science"
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

    preexec = final_text.index("PASS_PRECONTROLLER_ATTACHED_AUTHORITY")
    gpu_preflight = final_text.index("T4 x2 preflight could not query nvidia-smi")
    bootstrap = final_text.index("BOOTSTRAP = REPO / 'journal_extension/scripts/bootstrap_trackb_runtime.py'")
    operator_preflight = final_text.index("PASS_OPERATOR_PREFLIGHT")
    if preexec > gpu_preflight:
        raise ValidationError(
            "Notebook 01 must authenticate attached bytes/code before hardware preflight"
        )
    if gpu_preflight > bootstrap:
        raise ValidationError(
            "Notebook 01 must fail-fast on T4 x2 before runtime bootstrap"
        )
    if bootstrap > operator_preflight:
        raise ValidationError(
            "Notebook 01 must run publication capability probe only after locked runtime bootstrap"
        )

    probe = final_text.index("publication_probe = verify_kaggle_publication_capability(observed_owner)")
    pop_token = final_text.index("os.environ.pop('KAGGLE_API_TOKEN', None)", probe)
    controller = final_text.index(
        "CONTROLLER = REPO / 'journal_extension/scripts/trackb_v4_execute_attached.py'"
    )
    if probe > pop_token:
        raise ValidationError(
            "Notebook 01 publication probe must complete before credential scrubbing"
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
    guard_hotfix_source = (
        repo / "journal_extension/operator_hotfixes/trackb_v5_v1_guard_hotfix.py"
    ).read_text(encoding="utf-8")

    require_all(
        materializer_source,
        (
            "authoritative external cohorts — fail-fast sealed acquisition",
            "expected_source_manifest_sha256",
            "role_content_identity",
            "_role_content_identity",
            "verify_kaggle_published_file_roundtrip",
            "DEFERRED_TO_ATTACHED_FULL_BYTE_VERIFICATION",
            "NOTEBOOK_01_FULL_ATTACHED_ROLE_CONTENT_IDENTITY",
        ),
        "materializer",
    )
    deprecated_publication_tokens = (
        "_verify_published_archive_roundtrip",
        "shutil.rmtree(infra_root",
        "shutil.rmtree(external_root",
    )
    for token in deprecated_publication_tokens:
        if token in materializer_source:
            raise ValidationError(
                f"materializer retains deprecated async-unsafe publication primitive: {token}"
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
            "verify_kaggle_published_file_roundtrip",
            "BOUND_MANIFEST_ONLY_PENDING_ATTACHED_BYTE_VERIFICATION",
            "allow_version",
        ),
        "Track-B operations",
    )
    require_all(
        guard_hotfix_source,
        (
            "TRACKB_V5_V1_GUARD_FALSE_POSITIVE_FIX_v1",
            "v1_test_image_bytes_accessed",
            "V1_TRAIN_VAL_ONLY",
            "maximum_evidence_grade",
            "EXT-S",
            "_path_is_forbidden",
            "value is not False",
            "trackb_r07.assert_no_v1_test_surface",
            "_optimized_topk_cosine_neighbors",
            "_observable_encode_audit_features",
            "_stream_science_command",
            "DINO top-k",
            "science_subprocess_live_streaming",
        ),
        "Track-B V1 guard operator hotfix",
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
