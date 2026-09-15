from __future__ import annotations

import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    G2A_ACCOUNT_PROFILES,
    MASTER_WAIT_MAX_SECONDS,
    SCIENCE_SHA,
    OperatorError,
    ensure_private_dataset,
    fetch_public_file,
    g2a_public_run_id,
    kaggle_safe_dataset_slug,
    load_json,
    operator_runtime_head,
    publish_public_files,
    sanitized_child_env,
    sha256_file,
    validate_g2a_summary,
    validate_private_locators_with_science,
    wait_kaggle_dataset_ready,
    write_json,
)

REQUIRED_G2A = {"CAL-EFFB0", "CAL-CNXTT", "CAL-MNV4-LOGITS", "CAL-MNV4-FEATURE", "CAL-R13"}
G2A_STATUS_SCHEMA_VERSION = "1.0"
G2A_FAILURE_CODE = "ACCOUNT_G2A_QUALIFICATION_FAILED"


def publish_json(repo: Path, run_id: str, path: Path) -> str:
    return publish_public_files(repo, run_id, [path])


def g2a_summary_filename(calibration_id: str) -> str:
    return f"{calibration_id}_SUMMARY.json"


def public_run_id_for_g2a_account(account_id: str) -> str:
    return f"TRACKA-V12-G2A-{account_id}-{SCIENCE_SHA[:12]}"


def g2a_status_filename(account_id: str) -> str:
    return f"TRACKA_V12_G2A_{account_id}_STATUS.json"


def account_for_calibration(calibration_id: str) -> str:
    matches = [
        account_id
        for account_id, rows in G2A_ACCOUNT_PROFILES.items()
        if any(row[0] == calibration_id for row in rows)
    ]
    if len(matches) != 1:
        raise OperatorError(f"G2A calibration ownership must be unique: {calibration_id} -> {matches}")
    return matches[0]


def write_account_status(
    path: Path,
    *,
    account_id: str,
    status: str,
    g1a_seal_sha256: str,
    failure_code: str | None = None,
) -> dict:
    if status not in {"PREPARING", "READY", "FAILED"}:
        raise OperatorError(f"invalid G2A account status: {status}")
    if status == "FAILED" and failure_code != G2A_FAILURE_CODE:
        raise OperatorError("FAILED G2A account status requires frozen non-sensitive failure code")
    if status != "FAILED" and failure_code is not None:
        raise OperatorError(f"{status} G2A account status must not carry a failure code")
    payload = {
        "schema_version": G2A_STATUS_SCHEMA_VERSION,
        "stage": "TRACKA_V12_G2A_ACCOUNT_STATUS",
        "status": status,
        "account_id": account_id,
        "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(),
        "g1a_seal_sha256": g1a_seal_sha256,
        "failure_code": failure_code,
        "science_authorized": False,
        "protected_test_accessed": False,
        "external_surface_accessed": False,
    }
    write_json(path, payload)
    return payload


def _publish_account_status(
    repo: Path,
    path: Path,
    *,
    account_id: str,
    status: str,
    g1a_seal_sha256: str,
    failure_code: str | None = None,
) -> None:
    write_account_status(
        path,
        account_id=account_id,
        status=status,
        g1a_seal_sha256=g1a_seal_sha256,
        failure_code=failure_code,
    )
    publish_json(repo, public_run_id_for_g2a_account(account_id), path)


def _publish_failed_status_best_effort(
    repo: Path,
    path: Path,
    *,
    account_id: str,
    g1a_seal_sha256: str,
) -> None:
    write_account_status(
        path,
        account_id=account_id,
        status="FAILED",
        g1a_seal_sha256=g1a_seal_sha256,
        failure_code=G2A_FAILURE_CODE,
    )
    try:
        publish_json(repo, public_run_id_for_g2a_account(account_id), path)
    except Exception as publish_exc:
        print(
            "G2A_FAIL_STATUS_PUBLICATION_FAILED "
            f"account={account_id} error={type(publish_exc).__name__}; preserving original failure.",
            flush=True,
        )


def fetch_existing_g2a(repo: Path, calibration_id: str, destination: Path, *, expected_g1a_sha: str) -> dict | None:
    path = fetch_public_file(repo, g2a_public_run_id(calibration_id), g2a_summary_filename(calibration_id), destination)
    if path is None:
        return None
    return validate_g2a_summary(repo, path, expected_g1a_sha=expected_g1a_sha)


def g2a_profile_command(
    *,
    repo: Path,
    calibration_id: str,
    experiment_id: str,
    slot_id: str,
    manifest: Path,
    class_map: Path,
    image_root: Path,
    g1a_bundle: Path,
    output_dir: Path,
    summary_path: Path,
    durable_locator: str,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "journal_extension/scripts/qualify_tracka_v12_profile_v122.py"),
        "--repo-root",
        str(repo),
        "--calibration-id",
        calibration_id,
        "--experiment-id",
        experiment_id,
        "--manifest",
        str(manifest),
        "--class-map",
        str(class_map),
        "--image-root",
        str(image_root),
        "--g1a-bundle",
        str(g1a_bundle),
        "--run-id",
        f"TRACKA-V12-{calibration_id}-{SCIENCE_SHA[:12]}-A01",
        "--slot-id",
        slot_id,
        "--source-git-commit",
        SCIENCE_SHA,
        "--output-dir",
        str(output_dir),
        "--mode",
        "calibration",
        "--num-workers",
        "4",
        "--checkpoint-every-steps",
        "250",
        "--session-hard-limit-seconds",
        str(12 * 3600),
        "--finalization-margin-seconds",
        "3600",
        "--min-free-gb",
        "5",
        "--durable-store-kind",
        "kaggle-dataset",
        "--durable-store-locator",
        durable_locator,
        "--durable-required",
        "--initial-steps",
        "24",
        "--resume-steps",
        "8",
        "--summary-out",
        str(summary_path),
    ]


def ensure_account_g2a(
    repo: Path,
    *,
    account_id: str,
    username: str,
    kaggle_env: dict[str, str],
    manifest: Path,
    class_map: Path,
    image_root: Path,
    g1a_bundle: Path,
    g1a_seal: dict,
    master_root: Path,
    global_clock: float,
) -> dict[str, Path]:
    profiles = list(G2A_ACCOUNT_PROFILES[account_id])
    g2_root = master_root / "g2a"
    g2_root.mkdir(parents=True, exist_ok=True)
    status_path = g2_root / g2a_status_filename(account_id)
    g1a_sha = str(g1a_seal["g1a_seal_sha256"])
    _publish_account_status(
        repo,
        status_path,
        account_id=account_id,
        status="PREPARING",
        g1a_seal_sha256=g1a_sha,
    )

    try:
        summary_paths: dict[str, Path] = {}
        missing = []
        for calibration_id, experiment_id, slot_id, gpu_index in profiles:
            destination = g2_root / calibration_id / g2a_summary_filename(calibration_id)
            existing = fetch_existing_g2a(
                repo,
                calibration_id,
                destination,
                expected_g1a_sha=g1a_sha,
            )
            if existing is not None:
                print(f"{calibration_id}: reusing validated published G2A summary.")
                summary_paths[calibration_id] = destination
            else:
                missing.append((calibration_id, experiment_id, slot_id, gpu_index))

        def run_one(row):
            calibration_id, experiment_id, slot_id, gpu_index = row
            profile_root = g2_root / calibration_id
            profile_root.mkdir(parents=True, exist_ok=True)
            output_dir = profile_root / "run"
            summary_path = profile_root / g2a_summary_filename(calibration_id)
            shutil.rmtree(output_dir, ignore_errors=True)
            locator = f"{username}/{kaggle_safe_dataset_slug('cropcop-g2a', calibration_id)}"
            ensure_private_dataset(locator, env=kaggle_env)
            wait_kaggle_dataset_ready(locator, env=kaggle_env)
            validate_private_locators_with_science(repo, {calibration_id: locator}, env=kaggle_env)
            env = sanitized_child_env(gpu_index=gpu_index, global_clock=global_clock)
            env["KAGGLE_USERNAME"] = username
            env["KAGGLE_KEY"] = kaggle_env["KAGGLE_KEY"]
            command = g2a_profile_command(
                repo=repo,
                calibration_id=calibration_id,
                experiment_id=experiment_id,
                slot_id=slot_id,
                manifest=manifest,
                class_map=class_map,
                image_root=image_root,
                g1a_bundle=g1a_bundle,
                output_dir=output_dir,
                summary_path=summary_path,
                durable_locator=locator,
            )
            log_path = profile_root / "console.log"
            with log_path.open("w", encoding="utf-8") as log:
                cp = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            if cp.returncode != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise OperatorError(f"{calibration_id} failed rc={cp.returncode}\n{tail}")
            validate_g2a_summary(repo, summary_path, expected_g1a_sha=g1a_sha)
            publish_public_files(repo, g2a_public_run_id(calibration_id), [summary_path])
            return calibration_id, summary_path

        if missing:
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures = [pool.submit(run_one, row) for row in missing]
                for future in as_completed(futures):
                    calibration_id, path = future.result()
                    summary_paths[calibration_id] = path

        for calibration_id, *_ in profiles:
            destination = g2_root / calibration_id / g2a_summary_filename(calibration_id)
            canonical = fetch_existing_g2a(
                repo,
                calibration_id,
                destination,
                expected_g1a_sha=g1a_sha,
            )
            if canonical is None:
                raise OperatorError(f"{calibration_id} did not become canonical GitHub evidence")
            summary_paths[calibration_id] = destination

        report = {
            "schema_version": "1.0",
            "stage": "TRACKA_V12_G2A_ACCOUNT_MASTER",
            "status": "PASS",
            "account_id": account_id,
            "science_source_sha": SCIENCE_SHA,
            "operator_runtime_sha": operator_runtime_head(),
            "g1a_seal_sha256": g1a_sha,
            "calibrations": {
                cid: {"summary_sha256": sha256_file(path), "public_evidence_run_id": g2a_public_run_id(cid)}
                for cid, path in sorted(summary_paths.items())
            },
            "science_authorized": False,
        }
        report_path = g2_root / f"TRACKA_V12_G2A_{account_id}_PUBLIC_REPORT.json"
        write_json(report_path, report)
        publish_json(repo, public_run_id_for_g2a_account(account_id), report_path)
        _publish_account_status(
            repo,
            status_path,
            account_id=account_id,
            status="READY",
            g1a_seal_sha256=g1a_sha,
        )
        return summary_paths
    except BaseException:
        _publish_failed_status_best_effort(
            repo,
            status_path,
            account_id=account_id,
            g1a_seal_sha256=g1a_sha,
        )
        raise


def _fetch_current_account_status(
    repo: Path,
    *,
    account_id: str,
    g1a_sha: str,
    destination: Path,
) -> dict | None:
    path = fetch_public_file(
        repo,
        public_run_id_for_g2a_account(account_id),
        g2a_status_filename(account_id),
        destination,
    )
    if path is None:
        return None
    payload = load_json(path)
    if (
        payload.get("science_source_sha") != SCIENCE_SHA
        or payload.get("operator_runtime_sha") != operator_runtime_head()
        or payload.get("g1a_seal_sha256") != g1a_sha
    ):
        return None
    return payload


def _wait_for_calibration_or_failure(
    repo: Path,
    *,
    calibration_id: str,
    g1a_sha: str,
    destination: Path,
) -> Path:
    account_id = account_for_calibration(calibration_id)
    status_destination = destination.parent / g2a_status_filename(account_id)
    started = time.monotonic()
    while True:
        existing = fetch_existing_g2a(
            repo,
            calibration_id,
            destination,
            expected_g1a_sha=g1a_sha,
        )
        if existing is not None:
            return destination

        status = _fetch_current_account_status(
            repo,
            account_id=account_id,
            g1a_sha=g1a_sha,
            destination=status_destination,
        )
        if status and status.get("status") == "FAILED":
            if status.get("failure_code") != G2A_FAILURE_CODE:
                raise OperatorError(f"{account_id} G2A failure marker has invalid failure code")
            raise OperatorError(
                f"{account_id} reported G2A qualification failure for the exact current G1A/runtime; "
                f"K1 will not wait for {calibration_id}. Repair that account and rerun the masters fresh."
            )
        if time.monotonic() - started >= MASTER_WAIT_MAX_SECONDS:
            raise TimeoutError(f"timed out waiting for canonical G2A evidence: {calibration_id}")
        time.sleep(30)


def collect_all_g2a(repo: Path, *, g1a_seal: dict, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    g1a_sha = str(g1a_seal["g1a_seal_sha256"])
    for calibration_id in sorted(REQUIRED_G2A):
        path = destination / g2a_summary_filename(calibration_id)
        _wait_for_calibration_or_failure(
            repo,
            calibration_id=calibration_id,
            g1a_sha=g1a_sha,
            destination=path,
        )
        validate_g2a_summary(repo, path, expected_g1a_sha=g1a_sha)
        result[calibration_id] = path
    return result


def owner_from_summary(path: Path) -> str:
    payload = load_json(path)
    locator = str((payload.get("durability") or {}).get("locator", ""))
    owner, sep, slug = locator.partition("/")
    if not sep or not owner or not slug:
        raise OperatorError(f"invalid G2A durability locator in {path.name}: {locator}")
    return owner
