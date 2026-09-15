from __future__ import annotations

from pathlib import Path

from tracka_v12_kaggle_operator_v8 import OperatorError, control_public_run_id, fetch_public_bundle
from master_control_v8 import CONTROL_FILES, validate_control_bundle
from master_g2a_v8 import (
    G2A_FAILURE_CODE,
    REQUIRED_G2A,
    _fetch_current_account_status,
    account_for_calibration,
    fetch_existing_g2a,
    g2a_status_filename,
    validate_g2a_summary,
)


def try_collect_all_g2a(
    repo: Path,
    *,
    g1a_seal: dict,
    destination: Path,
) -> dict[str, Path] | None:
    """Return all five canonical G2A summaries only when already available.

    Missing-but-not-failed peer evidence is a controlled continuation condition,
    not a reason to burn the rest of a Kaggle session polling.
    """
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
