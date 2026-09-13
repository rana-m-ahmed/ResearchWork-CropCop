from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tracka_v12_kaggle_operator_v2 import (
    G2A_ACCOUNT_PROFILES,
    SCIENCE_SHA,
    assert_clean_science_checkout,
    assert_kaggle_paths,
    assert_python_version,
    discover_g1a_bundle,
    ensure_private_dataset,
    ensure_science_checkout,
    install_locked_stack,
    load_json,
    load_kaggle_credentials,
    resolve_frozen_dataset,
    resolve_image_root,
    sanitized_child_env,
    sha256_file,
    slugify,
    validate_private_locators_with_science,
    verify_locked_stack,
    write_json,
)


def gpu_names() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def validate_t4x2(names: list[str]) -> None:
    if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
        raise RuntimeError(f"G2A account requires qualified T4x2; observed {names}")


def profile_command(
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
        "--repo-root", str(repo),
        "--calibration-id", calibration_id,
        "--experiment-id", experiment_id,
        "--manifest", str(manifest),
        "--class-map", str(class_map),
        "--image-root", str(image_root),
        "--g1a-bundle", str(g1a_bundle),
        "--run-id", f"G2A-{calibration_id}-{SCIENCE_SHA[:12]}-A01",
        "--slot-id", slot_id,
        "--source-git-commit", SCIENCE_SHA,
        "--output-dir", str(output_dir),
        "--mode", "calibration",
        "--num-workers", "4",
        "--checkpoint-every-steps", "250",
        "--session-hard-limit-seconds", str(12 * 3600),
        "--finalization-margin-seconds", "3600",
        "--min-free-gb", "5",
        "--durable-store-kind", "kaggle-dataset",
        "--durable-store-locator", durable_locator,
        "--durable-required",
        "--initial-steps", "24",
        "--resume-steps", "8",
        "--summary-out", str(summary_path),
    ]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in G2A_ACCOUNT_PROFILES:
        raise SystemExit("usage: g2a_account_driver.py K1|K2|K3")
    account_id = sys.argv[1]
    profiles = G2A_ACCOUNT_PROFILES[account_id]

    assert_kaggle_paths()
    assert_python_version()

    output_root = Path(f"/kaggle/working/TRACKA_V12_G2A_{account_id}")
    if output_root.exists():
        raise RuntimeError(
            f"{output_root} already exists. G2A calibration is immutable; use a fresh Kaggle session "
            "rather than mixing partial attempts."
        )
    output_root.mkdir(parents=True)

    repo = ensure_science_checkout()
    install_locked_stack(repo)
    stack = verify_locked_stack(repo)
    assert_clean_science_checkout(repo)

    names = gpu_names()
    validate_t4x2(names)

    manifest, class_map = resolve_frozen_dataset(
        manifest_override=os.environ.get("CROPCOP_MANIFEST", ""),
        class_map_override=os.environ.get("CROPCOP_CLASS_MAP", ""),
    )
    image_root = resolve_image_root(
        manifest,
        override=os.environ.get("CROPCOP_IMAGE_ROOT", ""),
    )
    g1a_bundle = discover_g1a_bundle(
        override=os.environ.get("CROPCOP_G1A_BUNDLE", ""),
    )

    username, key = load_kaggle_credentials()
    kaggle_env = dict(os.environ)
    locators: dict[str, str] = {}
    for calibration_id, _experiment_id, _slot_id, _gpu_index in profiles:
        locator = f"{username}/cropcop-g2a-{slugify(calibration_id)}-{SCIENCE_SHA[:12]}"
        ensure_private_dataset(locator, env=kaggle_env)
        locators[calibration_id] = locator

    preflight = validate_private_locators_with_science(
        repo,
        {calibration_id: locators[calibration_id] for calibration_id, *_ in profiles},
        env=kaggle_env,
    )

    global_clock = time.monotonic()

    def run_profile(row: tuple[str, str, str, int]) -> tuple[str, int, Path, Path]:
        calibration_id, experiment_id, slot_id, gpu_index = row
        profile_root = output_root / calibration_id
        run_root = profile_root / "run"
        summary_path = profile_root / f"{calibration_id}_SUMMARY.json"
        log_path = profile_root / "console.log"
        profile_root.mkdir(parents=True, exist_ok=False)

        env = sanitized_child_env(gpu_index=gpu_index, global_clock=global_clock)
        env["KAGGLE_USERNAME"] = username
        env["KAGGLE_KEY"] = key
        cmd = profile_command(
            repo=repo,
            calibration_id=calibration_id,
            experiment_id=experiment_id,
            slot_id=slot_id,
            manifest=manifest,
            class_map=class_map,
            image_root=image_root,
            g1a_bundle=g1a_bundle,
            output_dir=run_root,
            summary_path=summary_path,
            durable_locator=locators[calibration_id],
        )
        with log_path.open("w", encoding="utf-8") as log:
            cp = subprocess.run(
                cmd,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return calibration_id, cp.returncode, summary_path, log_path

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(profiles)) as pool:
        futures = [pool.submit(run_profile, row) for row in profiles]
        for future in as_completed(futures):
            calibration_id, return_code, summary_path, log_path = future.result()
            results[calibration_id] = {
                "return_code": return_code,
                "summary_path": str(summary_path),
                "log_path": str(log_path),
            }

    failures = [cid for cid, row in results.items() if row["return_code"] != 0]
    if failures:
        tails = {
            cid: Path(results[cid]["log_path"]).read_text(encoding="utf-8", errors="replace")[-6000:]
            for cid in failures
        }
        raise RuntimeError("G2A profile failure(s):\n" + json.dumps(tails, indent=2))

    summaries = {}
    for calibration_id, *_ in profiles:
        path = Path(results[calibration_id]["summary_path"])
        payload = load_json(path)
        if payload.get("status") != "PASS":
            raise RuntimeError(f"{calibration_id} summary is not PASS")
        if payload.get("validation_enabled") is not False or payload.get("scientific_metric_computed") is not False:
            raise RuntimeError(f"{calibration_id} crossed the non-scientific boundary")
        if payload.get("source_git_commit") != SCIENCE_SHA:
            raise RuntimeError(f"{calibration_id} source SHA mismatch")
        if payload.get("durability", {}).get("locator") != locators[calibration_id]:
            raise RuntimeError(f"{calibration_id} durability locator mismatch")
        summaries[calibration_id] = payload

    report = {
        "schema_version": "1.0",
        "stage": "TRACKA_V12_G2A_ACCOUNT",
        "account_id": account_id,
        "status": "PASS",
        "science_source_sha": SCIENCE_SHA,
        "dependency_lock_sha256": stack["dependency_lock_sha256"],
        "kaggle_username": username,
        "gpu_names": names,
        "durability_preflight_status": preflight["status"],
        "calibrations": {
            calibration_id: {
                "summary_sha256": sha256_file(results[calibration_id]["summary_path"]),
                "durable_locator": locators[calibration_id],
                "validation_enabled": summaries[calibration_id]["validation_enabled"],
                "scientific_metric_computed": summaries[calibration_id]["scientific_metric_computed"],
            }
            for calibration_id, *_ in profiles
        },
    }
    write_json(output_root / f"G2A_{account_id}_OPERATOR_REPORT.json", report)
    archive = shutil.make_archive(
        f"/kaggle/working/TRACKA_V12_G2A_{account_id}_SUMMARIES",
        "zip",
        root_dir=output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Summary transfer archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
