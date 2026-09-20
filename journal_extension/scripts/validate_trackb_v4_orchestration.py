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

    expected_snapshot = "13b4d8fa19280f7ca94462e852a2c51dda416340"
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
    if "/kaggle/tmp/trackb_v4_materialization" not in readiness_text:
        raise ValidationError("readiness notebook must keep heavy materialization in ephemeral storage")
    if "trackb_v4_materialize.py" not in readiness_text:
        raise ValidationError("readiness notebook does not invoke the frozen materializer")
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
