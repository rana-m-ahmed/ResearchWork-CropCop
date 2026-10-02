from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.envelope import (
    EnvelopeError,
    child_environment,
    close_child_log,
    gpu_inventory,
    gpu_telemetry,
    gracefully_finalize_process_groups,
    launch_process,
    require_parent_batch,
    validate_t4x2_inventory,
)
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.secondary import operational_worker_count, PRINCIPAL_SCIENCE_SOURCE_SHA, load_json, validate_backfill_envelope_config
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source

EXPORTER = ROOT / "journal_extension/scripts/export_principal_validation.py"
ENVELOPES = ROOT / "journal_extension/kaggle/envelopes"


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvelopeError(f"required environment variable missing: {name}")
    return value


def envelope_path() -> Path:
    choice = req("CROPCOP_VALIDATION_BACKFILL")
    mapping = {
        "S1": "VAL_BACKFILL_S1_T4X2.json",
        "S2": "VAL_BACKFILL_S2_T4X2.json",
        "S3": "VAL_BACKFILL_S3_T4X2.json",
        "VAL-BACKFILL-S1-T4X2-V1": "VAL_BACKFILL_S1_T4X2.json",
        "VAL-BACKFILL-S2-T4X2-V1": "VAL_BACKFILL_S2_T4X2.json",
        "VAL-BACKFILL-S3-T4X2-V1": "VAL_BACKFILL_S3_T4X2.json",
    }
    if choice not in mapping:
        raise EnvelopeError("CROPCOP_VALIDATION_BACKFILL must be S1, S2 or S3")
    return ENVELOPES / mapping[choice]


def run_id(experiment_id: str) -> str:
    return f"JE-{experiment_id}-{PRINCIPAL_SCIENCE_SOURCE_SHA[:12]}-A01"


def git_env() -> tuple[dict[str, str], tempfile.TemporaryDirectory]:
    token = os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvelopeError("GitHub token required to retrieve/publish principal run evidence")
    td = tempfile.TemporaryDirectory()
    askpass = Path(td.name) / "askpass.py"
    askpass.write_text(
        "#!/usr/bin/env python3\nimport os,sys\np=(sys.argv[1] if len(sys.argv)>1 else '').lower()\n"
        "print('x-access-token' if 'username' in p else os.environ['CROPCOP_GITHUB_TOKEN'])\n",
        encoding="utf-8",
    )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["CROPCOP_GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS_REQUIRE"] = "force"
    return env, td


def fetch_run_record(rid: str, destination: Path) -> dict:
    env, td = git_env()
    try:
        branch = f"run-evidence/{rid}"
        remote = f"refs/remotes/origin/{branch}"
        cp = subprocess.run(
            ["git", "fetch", "origin", f"refs/heads/{branch}:{remote}"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        if cp.returncode != 0:
            raise EnvelopeError(f"failed to fetch principal evidence branch for {rid}")
        repo_path = f"journal_extension/evidence/public/runs/{rid}/run_record.json"
        show = subprocess.run(
            ["git", "show", f"{remote}:{repo_path}"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        if show.returncode != 0:
            raise EnvelopeError(f"principal evidence branch lacks run_record.json for {rid}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(show.stdout, encoding="utf-8")
        record = load_json(destination)
        if record.get("run_id") != rid:
            raise EnvelopeError("fetched principal run record ID mismatch")
        return record
    finally:
        td.cleanup()


def exporter_args(record_path: Path, out: Path) -> list[str]:
    args = [
        sys.executable, str(EXPORTER),
        "--run-record", str(record_path),
        "--manifest", req("CROPCOP_MANIFEST"),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--image-root", req("CROPCOP_IMAGE_ROOT"),
        "--output-dir", str(out),
        "--row-id-column", req("CROPCOP_ROW_ID_COLUMN"),
        "--path-column", req("CROPCOP_PATH_COLUMN"),
        "--split-column", req("CROPCOP_SPLIT_COLUMN"),
        "--label-column", req("CROPCOP_LABEL_COLUMN"),
        "--num-workers", str(operational_worker_count(os.environ)),
    ]
    class_index = os.environ.get("CROPCOP_CLASS_INDEX_COLUMN", "").strip()
    if class_index:
        args += ["--class-index-column", class_index]
    return args


def main() -> int:
    require_parent_batch("principal-validation-backfill-dual")
    budget = SessionBudget.from_environment(require_global_clock=True)
    config = load_json(envelope_path())
    errors = validate_backfill_envelope_config(config)
    if errors:
        raise SystemExit("validation-backfill envelope invalid: " + "; ".join(errors))

    inventory = gpu_inventory()
    errors = validate_t4x2_inventory(inventory)
    if errors:
        raise SystemExit("T4X2 inventory invalid: " + "; ".join(errors))
    output_root = Path(os.environ.get("CROPCOP_OUTPUT_ROOT", "/kaggle/working/cropcop-principal-validation")).resolve()
    verify_clean_source(ROOT, authorized_source_sha=req("CROPCOP_SOURCE_GIT_COMMIT"), output_roots=[output_root])
    output_root.mkdir(parents=True, exist_ok=True)

    kaggle_user = req("KAGGLE_USERNAME").lower()
    records = {}
    paths = {}
    for child in config["children"]:
        rid = run_id(child["experiment_id"])
        path = output_root / "records" / f"{rid}.json"
        record = fetch_run_record(rid, path)
        if record.get("source_git_commit") != PRINCIPAL_SCIENCE_SOURCE_SHA or record.get("status") != "PASS":
            raise SystemExit(f"principal record not terminal/frozen: {rid}")
        locator = str(record.get("durable_store", {}).get("locator") or "")
        if not locator.lower().startswith(kaggle_user + "/"):
            raise SystemExit(
                f"backfill must run on the Kaggle account that owns the durable checkpoint for {rid}; locator={locator}"
            )
        records[child["child_id"]] = record
        paths[child["child_id"]] = path

    running = {}
    outputs = {}
    for child in config["children"]:
        cid = child["child_id"]
        out = output_root / "children" / cid
        outputs[cid] = out
        env = child_environment(
            base_env=os.environ,
            envelope_id=config["envelope_id"],
            physical_slot=int(child["slot"]),
            output_root=out,
            g2_summaries_root=output_root / "unused",
            terminal_root=out / "terminal",
            workers=operational_worker_count(os.environ),
        )
        running[cid] = launch_process(
            child_id=cid,
            experiment_id=child["experiment_id"],
            run_id=records[cid]["run_id"],
            slot=int(child["slot"]),
            cmd=exporter_args(paths[cid], out),
            env=env,
            cwd=ROOT,
            log_path=out / "console.log",
        )

    telemetry = output_root / "GPU_TELEMETRY.jsonl"
    stop = {"value": False}
    old_handlers = {}
    def on_signal(_sig, _frame):
        stop["value"] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, on_signal)
    try:
        while any(c.process.poll() is None for c in running.values()):
            with telemetry.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp_epoch": time.time(), **gpu_telemetry()}, sort_keys=True) + "\n")
            if stop["value"] or budget.should_finalize(estimated_checkpoint_seconds=0, estimated_sync_seconds=120):
                gracefully_finalize_process_groups(list(running.values()), grace_seconds=180.0)
                break
            time.sleep(2.0)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for child in running.values():
            if child.process.poll() is None:
                gracefully_finalize_process_groups([child], grace_seconds=60.0)
            close_child_log(child)

    results = {}
    for child in config["children"]:
        cid = child["child_id"]
        out = outputs[cid]
        ev = out / "validation_evidence.json"
        preds = out / "validation_predictions.jsonl"
        rc = running[cid].process.poll()
        if rc != 0 or not ev.is_file() or not preds.is_file():
            results[cid] = {"status": "FAIL", "returncode": rc}
            continue
        evidence = load_json(ev)
        if evidence.get("status") != "PASS" or evidence.get("training_performed") is not False:
            results[cid] = {"status": "FAIL", "returncode": rc, "reason": "invalid validation evidence"}
            continue
        rid = records[cid]["run_id"]
        branch = publish_to_github_branch(
            repo_dir=ROOT,
            source_git_sha=PRINCIPAL_SCIENCE_SOURCE_SHA,
            run_id=rid,
            files=[ev, preds],
        )
        results[cid] = {
            "status": "PASS",
            "run_id": rid,
            "publication_branch": branch,
            "validation_predictions_sha256": evidence["validation_predictions_sha256"],
        }

    overall = "PASS" if results and all(v.get("status") == "PASS" for v in results.values()) else "FAIL"
    report = {
        "schema_version": "1.0",
        "status": overall,
        "envelope_id": config["envelope_id"],
        "principal_science_source_sha": PRINCIPAL_SCIENCE_SOURCE_SHA,
        "gpu_inventory": inventory,
        "children": results,
        "training_performed": False,
        "v1_test_accessed": False,
    }
    (output_root / "VALIDATION_BACKFILL_EVIDENCE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
