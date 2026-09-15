from __future__ import annotations

from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA,
    OperatorError,
    control_public_run_id,
    fetch_public_bundle,
    fetch_public_file,
    g1a_handoff_run_id,
    load_json,
    operator_runtime_head,
)
from master_control_v8 import CONTROL_FILES, validate_control_bundle
from master_g1a_v8 import G1A_FAILURE_CODE, G1A_HANDOFF_FILE
from master_g2a_v8 import (
    G2A_FAILURE_CODE,
    REQUIRED_G2A,
    _fetch_current_account_status,
    account_for_calibration,
    fetch_existing_g2a,
    g2a_status_filename,
    validate_g2a_summary,
)


def probe_current_g1a_handoff(repo: Path, *, destination: Path) -> dict | None:
    """Fetch the exact-runtime G1A handoff once; never poll.

    Missing or stale evidence is a controlled-continuation condition. A FAILED
    marker for this exact science/runtime identity remains fail-closed.
    """
    path = fetch_public_file(repo, g1a_handoff_run_id(), G1A_HANDOFF_FILE, destination)
    if path is None:
        return None
    payload = load_json(path)
    runtime_sha = operator_runtime_head()
    if payload.get("science_source_sha") != SCIENCE_SHA:
        return None
    if payload.get("operator_runtime_sha") != runtime_sha:
        return None
    status = payload.get("status")
    if status == "FAILED":
        if payload.get("failure_code") != G1A_FAILURE_CODE:
            raise OperatorError("current-runtime G1A FAILED handoff has invalid failure code")
        raise OperatorError("K1 reported canonical G1A failure for this exact science/runtime identity")
    if status != "READY":
        return None
    seal = str(payload.get("g1a_seal_sha256", "") or "")
    locator = str(payload.get("private_kaggle_dataset_locator", "") or "")
    if len(seal) != 64 or "/" not in locator:
        raise OperatorError("current-runtime READY G1A handoff is malformed")
    if payload.get("science_authorized") is not False:
        raise OperatorError("G1A handoff must remain non-authorizing")
    return payload


def try_collect_all_g2a(
    repo: Path,
    *,
    g1a_seal: dict,
    destination: Path,
) -> dict[str, Path] | None:
    """Return all five canonical G2A summaries only when already available."""
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    g1a_sha = str(g1a_seal["g1a_seal_sha256"])

    for calibration_id in sorted(REQUIRED_G2A):
        path = destination / f"{calibration_id}_SUMMARY.json"
        existing = fetch_existing_g2a(
            repo,
            calibration_id,
            path,
            expected_g1a_sha=g1a_sha,
        )
        if existing is None:
            owner = account_for_calibration(calibration_id)
            status_path = destination / g2a_status_filename(owner)
            status = _fetch_current_account_status(
                repo,
                account_id=owner,
                g1a_sha=g1a_sha,
                destination=status_path,
            )
            if status and status.get("status") == "FAILED":
                if status.get("failure_code") != G2A_FAILURE_CODE:
                    raise OperatorError(f"{owner} G2A failure marker has invalid failure code")
                raise OperatorError(
                    f"{owner} reported G2A qualification failure for exact current G1A/runtime"
                )
            return None
        validate_g2a_summary(repo, path, expected_g1a_sha=g1a_sha)
        result[calibration_id] = path

    return result


def try_acquire_control_worker(
    repo: Path,
    *,
    g1a_seal: dict,
    master_root: Path,
) -> tuple[Path, dict] | None:
    """Return canonical control immediately if published; otherwise continue later."""
    control_dir = master_root / "control"
    fetched = fetch_public_bundle(repo, control_public_run_id(), CONTROL_FILES, control_dir)
    if fetched is None:
        return None
    return control_dir, validate_control_bundle(repo, g1a_seal, control_dir)
