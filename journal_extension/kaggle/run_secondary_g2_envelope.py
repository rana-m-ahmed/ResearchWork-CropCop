from __future__ import annotations

import json
import os
import signal
import sys
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
from cropcop_je.g1_publication import ensure_private_target
from cropcop_je.persistence import validate_durable_access_plan, validate_durable_locator_template
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.secondary import load_json, operational_worker_count, validate_secondary_g1_bundle
from cropcop_je.secondary_g2 import (
    REQUIRED_SECONDARY_CALIBRATIONS,
    run_checkpoint_contract_probe,
    validate_secondary_calibration_summary,
    validate_secondary_g2_barrier_object,
    write_secondary_g2_barrier,
)
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source

CALIBRATE = ROOT / "journal_extension/scripts/calibrate_secondary.py"
CALIBRATION_QUEUE = (
    ("CAL-MNV4-DIRECT", "K1"),
    ("CAL-MNV4-TEACHER", "K1"),
    ("CAL-EFFB0", "K2"),
    ("CAL-CNXTT", "K2"),
)


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvelopeError(f"required environment variable missing: {name}")
    return value


def calibration_run_id(cid: str, source_sha: str) -> str:
    return f"SEC-{cid}-{source_sha[:12]}"


def common_args(source_sha: str, lane: str, g1: str, out: Path, cid: str, durable_kind: str, durable_locator: str) -> list[str]:
    args = [
        sys.executable, str(CALIBRATE),
        "--calibration-id", cid,
        "--repo-root", str(ROOT),
        "--manifest", req("CROPCOP_MANIFEST"),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--image-root", req("CROPCOP_IMAGE_ROOT"),
        "--secondary-g1-bundle", g1,
        "--source-git-commit", source_sha,
        "--lane-id", lane,
        "--output-dir", str(out),
        "--row-id-column", req("CROPCOP_ROW_ID_COLUMN"),
        "--path-column", req("CROPCOP_PATH_COLUMN"),
        "--split-column", req("CROPCOP_SPLIT_COLUMN"),
        "--label-column", req("CROPCOP_LABEL_COLUMN"),
        "--num-workers", str(operational_worker_count(os.environ)),
        "--durable-store-kind", durable_kind,
        "--durable-store-locator", durable_locator,
        "--durable-required",
    ]
    class_index = os.environ.get("CROPCOP_CLASS_INDEX_COLUMN", "").strip()
    if class_index:
        args += ["--class-index-column", class_index]
    args += [
        "--train-split-value", os.environ.get("CROPCOP_TRAIN_SPLIT_VALUE", "train"),
        "--val-split-value", os.environ.get("CROPCOP_VAL_SPLIT_VALUE", "val"),
    ]
    return args


def publish(source_sha: str, run_id: str, files: list[Path]) -> str:
    if not (os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        raise EnvelopeError("GitHub token required for secondary G2 public evidence publication")
    return publish_to_github_branch(repo_dir=ROOT, source_git_sha=source_sha, run_id=run_id, files=files)


def main() -> int:
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    if len(source_sha) != 40:
        raise SystemExit("CROPCOP_SOURCE_GIT_COMMIT must be an exact 40-character commit SHA")
    require_parent_batch("secondary-calibration-dual")
    budget = SessionBudget.from_environment(require_global_clock=True)

    g1_root = Path(req("CROPCOP_SECONDARY_G1_INPUT_ROOT")).resolve()
    seal, errors = validate_secondary_g1_bundle(g1_root)
    if errors:
        raise SystemExit("secondary G1 validation failed: " + "; ".join(errors))
    if seal.get("source_git_sha") != source_sha:
        raise SystemExit("secondary G1 source differs from calibration source")

    output_root = Path(os.environ.get("CROPCOP_OUTPUT_ROOT", "/kaggle/working/cropcop-secondary-g2")).resolve()
    verify_clean_source(ROOT, authorized_source_sha=source_sha, output_roots=[output_root, g1_root])
    inventory = gpu_inventory()
    inv_errors = validate_t4x2_inventory(inventory)
    if inv_errors:
        raise SystemExit("T4X2 inventory validation failed: " + "; ".join(inv_errors))
    output_root.mkdir(parents=True, exist_ok=True)

    durable_kind = req("CROPCOP_DURABLE_STORE_KIND")
    if durable_kind != "kaggle-dataset":
        raise SystemExit("real secondary G2 qualification requires kaggle-dataset durability")
    template = req("CROPCOP_DURABLE_LOCATOR_TEMPLATE")
    cal_run_ids = [calibration_run_id(cid, source_sha) for cid in REQUIRED_SECONDARY_CALIBRATIONS]
    locators = validate_durable_locator_template(durable_kind, template, cal_run_ids)
    create_policy = os.environ.get("CROPCOP_SECONDARY_ALLOW_CREATE_PRIVATE_DATASETS", "0").strip()
    if create_policy not in {"0", "1"}:
        raise SystemExit("CROPCOP_SECONDARY_ALLOW_CREATE_PRIVATE_DATASETS must be 0 or 1")
    if create_policy == "1":
        target_env = dict(os.environ)
        target_env["CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET"] = "1"
        for locator in locators.values():
            ensure_private_target(locator, env=target_env)
    access = validate_durable_access_plan(durable_kind, {rid: locators[rid] for rid in cal_run_ids}, env=os.environ)
    if access.get("status") != "PASS":
        raise SystemExit("secondary G2 durable access failed: " + "; ".join(access.get("errors", [])))

    checkpoint_probe = run_checkpoint_contract_probe(output_root)
    if checkpoint_probe.get("status") != "PASS":
        raise SystemExit("secondary G2 checkpoint contract probe failed")

    publication_probe_path = output_root / "G2_PUBLICATION_IDEMPOTENCY_PROBE.json"
    publication_probe_path.write_text(json.dumps({
        "schema_version": "1.0", "status": "PASS", "source_git_sha": source_sha,
        "purpose": "publication_idempotency_only", "scientific_result_produced": False,
        "protected_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    probe_run_id = f"SEC-G2-PUBLISH-PROBE-{source_sha[:12]}"
    branch1 = publish(source_sha, probe_run_id, [publication_probe_path])
    branch2 = publish(source_sha, probe_run_id, [publication_probe_path])
    publication_idempotency = {
        "status": "PASS" if branch1 == branch2 else "FAIL",
        "branch": branch1,
        "second_publish_same_branch": branch1 == branch2,
        "scientific_execution_relaunched": False,
    }
    if publication_idempotency["status"] != "PASS":
        raise SystemExit("secondary G2 publication idempotency probe failed")

    telemetry = output_root / "GPU_TELEMETRY.jsonl"
    shared = output_root / "summaries"
    shared.mkdir(parents=True, exist_ok=True)
    pending = list(CALIBRATION_QUEUE)
    running: dict[str, object] = {}
    results: dict[str, dict] = {}

    def start(cid: str, lane: str, slot: int) -> None:
        child_root = output_root / cid
        rid = calibration_run_id(cid, source_sha)
        env = child_environment(base_env=os.environ, envelope_id="SEC-G2-T4X2-V2", physical_slot=slot,
                                output_root=child_root, g2_summaries_root=shared,
                                terminal_root=child_root / "terminal", workers=operational_worker_count(os.environ))
        running[cid] = launch_process(
            child_id=cid, experiment_id=cid, run_id=rid, slot=slot,
            cmd=common_args(source_sha, lane, str(g1_root), child_root, cid, durable_kind, locators[rid]),
            env=env, cwd=ROOT, log_path=child_root / "console.log",
        )

    for slot in (0, 1):
        cid, lane = pending.pop(0)
        start(cid, lane, slot)

    stop_requested = {"value": False}
    old_handlers = {}
    def on_signal(_signum, _frame):
        stop_requested["value"] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, on_signal)

    try:
        while running:
            with telemetry.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp_epoch": time.time(), **gpu_telemetry()}, sort_keys=True) + "\n")
            if stop_requested["value"] or budget.should_finalize(estimated_checkpoint_seconds=180, estimated_sync_seconds=240):
                gracefully_finalize_process_groups(list(running.values()), grace_seconds=min(900.0, max(300.0, budget.finalization_margin_seconds * 0.75)))
            terminal_ids = []
            freed_slots = []
            for cid, child in list(running.items()):
                rc = child.process.poll()
                if rc is None:
                    continue
                close_child_log(child)
                terminal_ids.append(cid)
                freed_slots.append(child.slot)
                summary_path = output_root / cid / "calibration_summary.json"
                if rc != 0 or not summary_path.is_file():
                    results[cid] = {"status": "FAIL", "returncode": rc}
                    continue
                summary = load_json(summary_path)
                errs = validate_secondary_calibration_summary(summary)
                results[cid] = {"status": "PASS" if not errs else "FAIL", "returncode": rc, "errors": errs}
                if not errs:
                    target = shared / f"{cid}.json"
                    target.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
                    publish(source_sha, calibration_run_id(cid, source_sha), [target])
            for cid in terminal_ids:
                del running[cid]
            if stop_requested["value"] or budget.should_finalize(estimated_checkpoint_seconds=180, estimated_sync_seconds=240):
                break
            for slot in sorted(freed_slots):
                if pending:
                    cid, lane = pending.pop(0)
                    start(cid, lane, slot)
            time.sleep(2.0)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for child in list(running.values()):
            if child.process.poll() is None:
                gracefully_finalize_process_groups([child], grace_seconds=120.0)
            close_child_log(child)

    required = set(REQUIRED_SECONDARY_CALIBRATIONS)
    passed = {cid for cid, row in results.items() if row.get("status") == "PASS"}
    if passed != required:
        evidence = {
            "schema_version": "1.0", "status": "FAIL_OR_INCOMPLETE", "source_git_sha": source_sha,
            "required_calibrations": sorted(required), "passed_calibrations": sorted(passed),
            "pending_calibrations": [cid for cid, _ in pending], "results": results,
            "gpu_inventory": inventory, "durable_access": access,
            "checkpoint_contract_probe": checkpoint_probe, "publication_idempotency": publication_idempotency,
        }
        (output_root / "SECONDARY_G2_ENVELOPE_EVIDENCE.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit("secondary G2 calibration envelope incomplete/failed: " + json.dumps(results, sort_keys=True))

    barrier_path = output_root / "SECONDARY_G2_CALIBRATION_BARRIER.json"
    summaries = [load_json(shared / f"{cid}.json") for cid in REQUIRED_SECONDARY_CALIBRATIONS]
    barrier = write_secondary_g2_barrier(barrier_path, summaries, checkpoint_contract_probe=checkpoint_probe,
                                         publication_idempotency=publication_idempotency)
    errors = validate_secondary_g2_barrier_object(barrier, expected_source_sha=source_sha,
                                                   expected_g1_seal_sha256=seal["secondary_g1_seal_sha256"])
    if errors:
        raise SystemExit("secondary G2 barrier failed: " + "; ".join(errors))

    evidence = {
        "schema_version": "2.0", "status": "PASS", "source_git_sha": source_sha,
        "secondary_g1_seal_sha256": seal["secondary_g1_seal_sha256"],
        "g2_barrier_sha256": barrier["barrier_sha256"], "calibrations": results,
        "gpu_inventory": inventory, "durable_access": access,
        "checkpoint_contract_probe": checkpoint_probe, "publication_idempotency": publication_idempotency,
        "model_construction_coverage": sorted(barrier["covered_secondary_experiment_ids"]),
        "v1_test_accessed": False, "external_protected_surface_accessed": False,
    }
    evidence_path = output_root / "SECONDARY_G2_ENVELOPE_EVIDENCE.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    publish(source_sha, f"SEC-G2-BARRIER-{source_sha[:12]}", [barrier_path, evidence_path, *sorted(shared.glob("*.json"))])
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
