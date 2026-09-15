from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import master_science_v8 as base
from tracka_v12_kaggle_operator_v8 import OperatorError

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SRC = RUNTIME_ROOT / "journal_extension" / "src"
PRISTINE_ENV = "CROPCOP_V13_PRISTINE_EXPERIMENTS"
SCIENCE_REPO_ENV = "CROPCOP_V13_SCIENCE_REPO"
BOOTSTRAP_README = "Private CropCop operational durability/handoff dataset. Do not make public.\n"
MAX_DIAGNOSTIC_CHARS = 32000
_V8_SCIENCE_COMMAND = base.science_command


def _private_state(locator: str, env: dict[str, str]) -> dict:
    if str(RUNTIME_SRC) not in sys.path:
        sys.path.insert(0, str(RUNTIME_SRC))
    from cropcop_je.g1_publication import preflight_private_target

    return preflight_private_target(locator, env=env)


def _version_number(state: dict) -> int:
    value = state.get("current_version_number")
    if value is None:
        raise OperatorError("scientific durability target has no current_version_number")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise OperatorError(f"invalid scientific durability current_version_number: {value!r}") from exc
    if version < 1:
        raise OperatorError(f"invalid scientific durability version number: {version}")
    return version


def _download_dataset(locator: str, destination: Path, env: dict[str, str]) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    cp = subprocess.run(
        ["kaggle", "datasets", "download", "-d", locator, "-p", str(destination), "--unzip", "-q"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()[-2000:]
        raise OperatorError(f"v13 bootstrap inspection download failed for {locator}: {detail}")


def _classify_downloaded_generation(root: Path, *, version: int, locator: str) -> str:
    marker = root / "durable_sync.json"
    index = root / "checkpoint_index.json"
    if marker.is_file() or index.is_file():
        if not marker.is_file() or not index.is_file():
            raise OperatorError(
                f"{locator}: partial durable recovery generation detected; refusing bootstrap "
                f"marker={marker.is_file()} index={index.is_file()}"
            )
        return "AUTO_RESUME"

    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "dataset-metadata.json"
    )
    if version != 1:
        raise OperatorError(
            f"{locator}: durability version {version} has no recovery marker/index; "
            "refusing a silent fresh scientific restart"
        )
    if files != ["README.txt"]:
        raise OperatorError(
            f"{locator}: initial durability generation has unexpected files {files}; "
            "refusing pristine-bootstrap classification"
        )
    readme = (root / "README.txt").read_text(encoding="utf-8")
    if readme != BOOTSTRAP_README:
        raise OperatorError(f"{locator}: bootstrap README bytes differ from the qualified initial dataset contract")
    return "PRISTINE_FIRST_RUN"


def classify_durable_target(locator: str, env: dict[str, str]) -> dict:
    state = _private_state(locator, env)
    version = _version_number(state)
    if version > 1:
        return {
            "locator": locator,
            "version": version,
            "mode": "AUTO_RESUME",
            "inspection": "metadata_version_gt_1",
        }
    with tempfile.TemporaryDirectory(dir="/kaggle/working" if Path("/kaggle/working").is_dir() else None) as td:
        root = Path(td)
        _download_dataset(locator, root, env)
        mode = _classify_downloaded_generation(root, version=version, locator=locator)
    return {
        "locator": locator,
        "version": version,
        "mode": mode,
        "inspection": "exact_initial_generation_download",
    }


def classify_account_durability(account_id: str, control: dict, env: dict[str, str]) -> tuple[set[str], list[dict]]:
    queues = control["scheduler"]["static_slot_queues"]
    durable_map = control["durable_map"]
    assigned: list[str] = []
    for slot_id in (f"{account_id}/GPU0", f"{account_id}/GPU1"):
        queue = queues.get(slot_id)
        if not isinstance(queue, list) or not queue:
            raise OperatorError(f"v13 bootstrap classification missing queue for {slot_id}")
        assigned.extend(str(value) for value in queue)
    if len(assigned) != len(set(assigned)):
        raise OperatorError(f"v13 account queue contains duplicate scientific states: {account_id}")

    rows: list[dict] = []
    pristine: set[str] = set()
    for experiment_id in assigned:
        locator = str(durable_map.get(experiment_id, ""))
        if not locator:
            raise OperatorError(f"v13 durable map missing {experiment_id}")
        row = {"experiment_id": experiment_id, **classify_durable_target(locator, env)}
        rows.append(row)
        if row["mode"] == "PRISTINE_FIRST_RUN":
            pristine.add(experiment_id)
    return pristine, rows


def science_command_v13(
    repo: Path,
    *,
    account_id: str,
    manifest: Path,
    class_map: Path,
    image_root: Path,
    g1a_bundle: Path,
    control_dir: Path,
    master_root: Path,
) -> list[str]:
    command = _V8_SCIENCE_COMMAND(
        repo,
        account_id=account_id,
        manifest=manifest,
        class_map=class_map,
        image_root=image_root,
        g1a_bundle=g1a_bundle,
        control_dir=control_dir,
        master_root=master_root,
    )
    command[1] = str(Path(__file__).resolve().parent / "science_account_runner_v13.py")
    return command


def _redact(text: str) -> str:
    patterns = (
        r"(?i)(KAGGLE_KEY\s*[=:]\s*)[^\s'\"\\]+",
        r"(?i)(GITHUB_TOKEN\s*[=:]\s*)[^\s'\"\\]+",
        r"(?i)(GH_TOKEN\s*[=:]\s*)[^\s'\"\\]+",
        r"(?i)(CROPCOP_GITHUB_TOKEN\s*[=:]\s*)[^\s'\"\\]+",
    )
    out = text
    for pattern in patterns:
        out = re.sub(pattern, r"\1[REDACTED]", out)
    return out


def _tail_text(path: Path) -> str:
    if not path.is_file():
        return "<console.log missing>"
    data = path.read_bytes()
    tail = data[-MAX_DIAGNOSTIC_CHARS:]
    return _redact(tail.decode("utf-8", errors="replace"))


def emit_failure_diagnostics(account_id: str, master_root: Path) -> None:
    summary_path = master_root / "science" / account_id / "ACCOUNT_EXECUTION_SUMMARY.json"
    if not summary_path.is_file():
        print(f"V13_FAILURE_DIAGNOSTIC account={account_id} summary_missing={summary_path}", flush=True)
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"V13_FAILURE_DIAGNOSTIC account={account_id} summary_unreadable={type(exc).__name__}: {exc}", flush=True)
        return

    allowed_root = (master_root / "science" / account_id).resolve()
    for slot_id, rows in (summary.get("slot_results") or {}).items():
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or row.get("return_code") in {None, 0}:
                continue
            experiment_id = str(row.get("experiment_id") or "UNKNOWN")
            raw_log = str(row.get("console_log") or "")
            log_path = Path(raw_log).resolve() if raw_log else allowed_root / experiment_id / "console.log"
            if log_path != allowed_root and allowed_root not in log_path.parents:
                print(
                    f"V13_FAILURE_DIAGNOSTIC account={account_id} slot={slot_id} experiment={experiment_id} "
                    "console_path_rejected=outside_account_root",
                    flush=True,
                )
                continue
            run_dir = log_path.parent
            lifecycle = {
                "run_owner": (run_dir / ".run_owner.json").is_file(),
                "run_record": (run_dir / "run_record.json").is_file(),
                "segments": (run_dir / "segments.jsonl").is_file(),
                "checkpoint_index": (run_dir / "private_checkpoints" / "checkpoint_index.json").is_file(),
                "console_bytes": log_path.stat().st_size if log_path.is_file() else 0,
            }
            print(
                "\n=== V13_SCIENTIFIC_CHILD_FAILURE "
                f"account={account_id} slot={slot_id} experiment={experiment_id} rc={row.get('return_code')} ===",
                flush=True,
            )
            print("V13_CHILD_LIFECYCLE " + json.dumps(lifecycle, sort_keys=True), flush=True)
            print(_tail_text(log_path), flush=True)
            print("=== V13_SCIENTIFIC_CHILD_FAILURE_END ===\n", flush=True)


def run_science(
    repo: Path,
    *,
    account_id: str,
    manifest: Path,
    class_map: Path,
    image_root: Path,
    g1a_bundle: Path,
    control_dir: Path,
    control: dict,
    master_root: Path,
) -> int:
    kaggle_env = dict(os.environ)
    username = str(kaggle_env.get("KAGGLE_USERNAME", "") or "").strip()
    key = str(kaggle_env.get("KAGGLE_KEY", "") or "").strip()
    if not username or not key:
        raise OperatorError("v13 science bootstrap classification requires authenticated Kaggle credentials")

    pristine, classification = classify_account_durability(account_id, control, kaggle_env)
    print(
        "V13_DURABILITY_BOOTSTRAP_CLASSIFICATION "
        + json.dumps({"account_id": account_id, "targets": classification}, sort_keys=True),
        flush=True,
    )

    previous_pristine = os.environ.get(PRISTINE_ENV)
    previous_repo = os.environ.get(SCIENCE_REPO_ENV)
    previous_command = base.science_command
    os.environ[PRISTINE_ENV] = json.dumps(sorted(pristine))
    os.environ[SCIENCE_REPO_ENV] = str(repo.resolve())
    base.science_command = science_command_v13
    try:
        try:
            return base.run_science(
                repo,
                account_id=account_id,
                manifest=manifest,
                class_map=class_map,
                image_root=image_root,
                g1a_bundle=g1a_bundle,
                control_dir=control_dir,
                control=control,
                master_root=master_root,
            )
        except BaseException:
            emit_failure_diagnostics(account_id, master_root)
            raise
    finally:
        base.science_command = previous_command
        if previous_pristine is None:
            os.environ.pop(PRISTINE_ENV, None)
        else:
            os.environ[PRISTINE_ENV] = previous_pristine
        if previous_repo is None:
            os.environ.pop(SCIENCE_REPO_ENV, None)
        else:
            os.environ[SCIENCE_REPO_ENV] = previous_repo
