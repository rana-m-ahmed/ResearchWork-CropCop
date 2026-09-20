from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


class ValidationError(RuntimeError):
    pass


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob_sha(repo: Path, path: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(repo), "hash-object", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


def notebook_text(path: Path) -> tuple[dict, str]:
    nb = load_json(path)
    if nb.get("nbformat") != 4:
        raise ValidationError(f"unexpected nbformat: {path}")
    text = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
    )
    return nb, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()

    lock = load_json(repo / "journal_extension/track_b_r07/TRACKB_ORCHESTRATION_LOCK_v4.json")
    mat = load_json(repo / "journal_extension/track_b_r07/TRACKB_INPUT_MATERIALIZATION_LOCK_v1.json")

    if lock.get("lock_id") != "TRACKB_ORCHESTRATION_LOCK_v4":
        raise ValidationError("unexpected v4 orchestration lock identity")
    if lock.get("scientific_change") is not False:
        raise ValidationError("v4 orchestration lock must explicitly declare no scientific change")
    if mat.get("protected_external_predictions_allowed") is not False:
        raise ValidationError("materialization lock must forbid protected external predictions")
    if mat.get("source_inputs", {}).get("attachment_mode") != "KAGGLE_ATTACHED_DATASETS_ONLY":
        raise ValidationError("readiness source-input mode drift")

    for path, expected in (lock.get("load_bearing_files") or {}).items():
        observed = git_blob_sha(repo, path)
        if observed != expected:
            raise ValidationError(
                f"v4 load-bearing blob mismatch: {path}: expected={expected}, observed={observed}"
            )

    readiness_path = repo / "journal_extension/kaggle/TrackB_00_Readiness_Materialization.ipynb"
    final_path = repo / "journal_extension/kaggle/TrackB_01_Final_Execution.ipynb"
    readiness, readiness_text = notebook_text(readiness_path)
    final, final_text = notebook_text(final_path)

    if len(readiness.get("cells", [])) != 5:
        raise ValidationError("readiness notebook must have exactly five cells")
    if len(final.get("cells", [])) != 6:
        raise ValidationError("final execution notebook must have exactly six cells")

    expected_snapshot = "4c2f49f2698ea321c04b1a3ce03f4e18b5ac3183"
    if f"SOURCE_COMMIT = '{expected_snapshot}'" not in readiness_text:
        raise ValidationError("readiness notebook is not pinned to the frozen v4 repository snapshot")
    for basename in (
        "cropcop-finalized-v8-11-2026-1",
        "sec-je-r07-cnxtt-context-s1-8904b100d223-a01",
        "cropcop-r07-cnxtt-context-s2-abce1197-56023042",
        "cropcop-r07-cnxtt-context-s3-f13ca687-56023042",
        "cropcop-secondary-g1-8904b100",
    ):
        if basename not in readiness_text:
            raise ValidationError(f"readiness notebook missing attached-source declaration: {basename}")
    if "OUT = Path('/kaggle/tmp/trackb_v4_materialization')" not in readiness_text:
        raise ValidationError("readiness notebook must launch heavy materialization in ephemeral storage")
    if "receipt_path = Path('/kaggle/tmp/trackb_v4_materialization')" not in readiness_text:
        raise ValidationError("readiness notebook receipt path must match ephemeral materialization root")
    if "OUT = WORK / 'trackb_v4_materialization'" in readiness_text:
        raise ValidationError("readiness notebook contains stale persistent materialization path")
    if "trackb_v4_materialize.py" not in readiness_text:
        raise ValidationError("readiness notebook does not invoke the frozen materializer")
    builder_source = (repo / "journal_extension/scripts/build_trackb_core_package.py").read_text(encoding="utf-8")
    materializer_source = (repo / "journal_extension/scripts/trackb_v4_materialize.py").read_text(encoding="utf-8")
    if '"--dino-factory-source-root", "repository/journal_extension/teacher_factory"' not in builder_source:
        raise ValidationError("Track-B core builder does not bind the sealed DINO factory to its dedicated source root")
    if "validate_teacher_factory_bundle(" not in materializer_source or "load_exact_teacher(" not in materializer_source:
        raise ValidationError("Track-B readiness must validate and instantiate the sealed DINO teacher during source qualification")
    historical_source = (repo / "journal_extension/scripts/build_trackb_historical_compare.py").read_text(encoding="utf-8")
    ops_source = (repo / "journal_extension/src/cropcop_je/trackb_r07_ops.py").read_text(encoding="utf-8")
    if "dino_batch_probe_size" not in historical_source or "len(dino_samples) != 64" not in historical_source:
        raise ValidationError("historical builder must smoke the real 64-image DINO batch before full indexing")
    if "shutil.disk_usage(output_root).free" not in historical_source or "_historical_output_budget_bytes()" not in historical_source:
        raise ValidationError("historical builder lacks output-filesystem disk preflight")
    if "max_members: int = 120000" not in ops_source or "insufficient disk before archive extraction" not in ops_source:
        raise ValidationError("Track-B archive extraction lacks bounded member/size/disk safeguards")
    if "Irish Potato source probe found missing checksum" not in ops_source or "download_urls_valid" not in ops_source:
        raise ValidationError("Track-B external transport preflight lacks URL/checksum/size hardening")
    if 'infra_manifest = load_json(infra_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")' not in materializer_source:
        raise ValidationError("Track-B publication must snapshot the infrastructure manifest before local cleanup")
    if materializer_source.index("shutil.rmtree(infra_root, ignore_errors=True)") > materializer_source.index('infra_pub["archive_roundtrip"] = _verify_published_archive_roundtrip('):
        raise ValidationError("Track-B infrastructure round-trip must release the local bundle before ZIP verification")
    if "source_qualification_sha256" not in readiness_text:
        raise ValidationError("readiness notebook must surface the complete source-qualification digest")
    if "CROPCOP_GITHUB_TOKEN" in readiness_text:
        raise ValidationError("readiness notebook must not require a GitHub credential")

    if "RUN_MODE = 'qualification'" not in final_text:
        raise ValidationError("final execution notebook must default to qualification")
    if "trackb_v4_execute_attached.py" not in final_text:
        raise ValidationError("final execution notebook does not invoke attached-input controller")
    if "TRACKB_INFRASTRUCTURE_BUNDLE.json" not in final_text or "TRACKB_EXTERNAL_BUNDLE.json" not in final_text:
        raise ValidationError("final notebook does not require the two paired v4 bundles")
    for forbidden in (
        "data.mendeley.com",
        "zenodo.org/api",
        "kaggle datasets download",
        "git clone",
        "run_trackb_r07_master.py",
    ):
        if forbidden in final_text:
            raise ValidationError(f"final execution notebook regressed to live acquisition/orchestration: {forbidden}")
    if "get_secret('CROPCOP_GITHUB_TOKEN')" in final_text:
        raise ValidationError("final scientific notebook must not request GitHub credentials")
    if "os.environ.pop('KAGGLE_API_TOKEN', None)" not in final_text:
        raise ValidationError("qualification mode must explicitly remove Kaggle API credentials")

    result = {
        "status": "PASS_TRACKB_V4_ORCHESTRATION_QA",
        "scientific_change": False,
        "supported_notebooks": lock["supported_operator_notebooks"],
        "qualification_secrets": lock["qualification"]["secrets_required"],
        "claim_github_token_required": lock["claim"]["github_token_required"],
        "materialization_attachment_mode": mat["source_inputs"]["attachment_mode"],
        "final_live_scientific_downloads": lock["qualification"]["live_scientific_input_downloads"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
