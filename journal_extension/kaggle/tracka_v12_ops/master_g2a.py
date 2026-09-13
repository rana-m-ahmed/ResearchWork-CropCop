from __future__ import annotations

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    G2A_ACCOUNT_PROFILES, SCIENCE_SHA, OperatorError, ensure_private_dataset, fetch_public_file,
    g2a_public_run_id, kaggle_safe_dataset_slug, load_json, operator_runtime_head,
    publish_public_files, sanitized_child_env, sha256_file, validate_g2a_summary,
    validate_private_locators_with_science, wait_for_public_file, wait_kaggle_dataset_ready, write_json,
)

REQUIRED_G2A = {"CAL-EFFB0", "CAL-CNXTT", "CAL-MNV4-LOGITS", "CAL-MNV4-FEATURE", "CAL-R13"}


def publish_json(repo: Path, run_id: str, path: Path) -> str:
    return publish_public_files(repo, run_id, [path])


def g2a_summary_filename(calibration_id: str) -> str:
    return f"{calibration_id}_SUMMARY.json"


def fetch_existing_g2a(repo: Path, calibration_id: str, destination: Path, *, expected_g1a_sha: str) -> dict | None:
    path = fetch_public_file(repo, g2a_public_run_id(calibration_id), g2a_summary_filename(calibration_id), destination)
    if path is None:
        return None
    return validate_g2a_summary(repo, path, expected_g1a_sha=expected_g1a_sha)


def g2a_profile_command(
    *, repo: Path, calibration_id: str, experiment_id: str, slot_id: str, manifest: Path,
    class_map: Path, image_root: Path, g1a_bundle: Path, output_dir: Path,
    summary_path: Path, durable_locator: str,
) -> list[str]:
    return [
        sys.executable, str(repo / "journal_extension/scripts/qualify_tracka_v12_profile_v122.py"),
        "--repo-root", str(repo), "--calibration-id", calibration_id, "--experiment-id", experiment_id,
        "--manifest", str(manifest), "--class-map", str(class_map), "--image-root", str(image_root),
        "--g1a-bundle", str(g1a_bundle), "--run-id", f"TRACKA-V12-{calibration_id}-{SCIENCE_SHA[:12]}-A01",
        "--slot-id", slot_id, "--source-git-commit", SCIENCE_SHA, "--output-dir", str(output_dir),
        "--mode", "calibration", "--num-workers", "4", "--checkpoint-every-steps", "250",
        "--session-hard-limit-seconds", str(12 * 3600), "--finalization-margin-seconds", "3600",
        "--min-free-gb", "5", "--durable-store-kind", "kaggle-dataset",
        "--durable-store-locator", durable_locator, "--durable-required",
        "--initial-steps", "24", "--resume-steps", "8", "--summary-out", str(summary_path),
    ]


def ensure_account_g2a(
    repo: Path, *, account_id: str, username: str, kaggle_env: dict[str, str], manifest: Path,
    class_map: Path, image_root: Path, g1a_bundle: Path, g1a_seal: dict,
    master_root: Path, global_clock: float,
) -> dict[str, Path]:
    profiles = list(G2A_ACCOUNT_PROFILES[account_id])
    g2_root = master_root / "g2a"
    g2_root.mkdir(parents=True, exist_ok=True)
    summary_paths: dict[str, Path] = {}
    missing = []
    for calibration_id, experiment_id, slot_id, gpu_index in profiles:
        destination = g2_root / calibration_id / g2a_summary_filename(calibration_id)
        existing = fetch_existing_g2a(
            repo, calibration_id, destination, expected_g1a_sha=g1a_seal["g1a_seal_sha256"],
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
            repo=repo, calibration_id=calibration_id, experiment_id=experiment_id, slot_id=slot_id,
            manifest=manifest, class_map=class_map, image_root=image_root, g1a_bundle=g1a_bundle,
            output_dir=output_dir, summary_path=summary_path, durable_locator=locator,
        )
        log_path = profile_root / "console.log"
        with log_path.open("w", encoding="utf-8") as log:
            cp = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        if cp.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise OperatorError(f"{calibration_id} failed rc={cp.returncode}\n{tail}")
        validate_g2a_summary(repo, summary_path, expected_g1a_sha=g1a_seal["g1a_seal_sha256"])
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
            repo, calibration_id, destination, expected_g1a_sha=g1a_seal["g1a_seal_sha256"],
        )
        if canonical is None:
            raise OperatorError(f"{calibration_id} did not become canonical GitHub evidence")
        summary_paths[calibration_id] = destination

    report = {
        "schema_version": "1.0", "stage": "TRACKA_V12_G2A_ACCOUNT_MASTER", "status": "PASS",
        "account_id": account_id, "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(), "g1a_seal_sha256": g1a_seal["g1a_seal_sha256"],
        "calibrations": {
            cid: {"summary_sha256": sha256_file(path), "public_evidence_run_id": g2a_public_run_id(cid)}
            for cid, path in sorted(summary_paths.items())
        },
        "science_authorized": False,
    }
    report_path = g2_root / f"TRACKA_V12_G2A_{account_id}_PUBLIC_REPORT.json"
    write_json(report_path, report)
    publish_json(repo, public_run_id_for_g2a_account(account_id), report_path)
    return summary_paths


def public_run_id_for_g2a_account(account_id: str) -> str:
    return f"TRACKA-V12-G2A-{account_id}-{SCIENCE_SHA[:12]}"


def collect_all_g2a(repo: Path, *, g1a_seal: dict, destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for calibration_id in sorted(REQUIRED_G2A):
        path = destination / g2a_summary_filename(calibration_id)
        wait_for_public_file(
            repo, g2a_public_run_id(calibration_id), g2a_summary_filename(calibration_id), path,
            predicate=lambda p, cid=calibration_id: p.get("status") == "PASS"
            and p.get("calibration_id") == cid and p.get("source_git_commit") == SCIENCE_SHA
            and p.get("g1a_seal_sha256") == g1a_seal["g1a_seal_sha256"],
        )
        validate_g2a_summary(repo, path, expected_g1a_sha=g1a_seal["g1a_seal_sha256"])
        result[calibration_id] = path
    return result


def owner_from_summary(path: Path) -> str:
    payload = load_json(path)
    locator = str((payload.get("durability") or {}).get("locator", ""))
    owner, sep, slug = locator.partition("/")
    if not sep or not owner or not slug:
        raise OperatorError(f"invalid G2A durability locator in {path.name}: {locator}")
    return owner
