from __future__ import annotations

import json
import os
import signal
import subprocess
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
from cropcop_je.secondary_g2 import validate_secondary_g2_barrier_object
from cropcop_je.g1_publication import ensure_private_target
from cropcop_je.persistence import validate_durable_access_plan, validate_durable_locator_template
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.secondary import operational_worker_count, load_json, validate_secondary_envelope_config, validate_secondary_g1_bundle
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source

RUNNER = ROOT / "journal_extension/scripts/run_secondary_training.py"
ENVELOPES = ROOT / "journal_extension/kaggle/envelopes"


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvelopeError(f"required environment variable missing: {name}")
    return value


def envelope_path() -> Path:
    choice = req("CROPCOP_SECONDARY_ENVELOPE")
    mapping = {
        "mechanism": "SEC_MECHANISM_T4X2.json",
        "context": "SEC_CONTEXT_T4X2.json",
        "SEC-MECHANISM-T4X2-V1": "SEC_MECHANISM_T4X2.json",
        "SEC-CONTEXT-T4X2-V1": "SEC_CONTEXT_T4X2.json",
    }
    if choice not in mapping:
        raise EnvelopeError("CROPCOP_SECONDARY_ENVELOPE must be mechanism or context")
    return ENVELOPES / mapping[choice]


def run_id(experiment_id: str, source_sha: str) -> str:
    key = "CROPCOP_RUN_ID_" + "".join(ch if ch.isalnum() else "_" for ch in experiment_id).upper()
    attempt = int(os.environ.get(key + "_ATTEMPT", "1"))
    if not 1 <= attempt <= 99:
        raise EnvelopeError(f"{key}_ATTEMPT must be 1..99")
    expected = f"JE-{experiment_id}-{source_sha[:12]}-A{attempt:02d}"
    explicit = os.environ.get(key, "").strip()
    if explicit and explicit != expected:
        raise EnvelopeError(
            f"explicit run ID in {key} is not bound to experiment/source/attempt; "
            f"expected {expected}"
        )
    return expected


def common_args(source_sha: str, lane: str, g1: Path, g2: Path, child_out: Path, eid: str, rid: str, durable_kind: str, durable_locator: str, resume_mode: str) -> list[str]:
    args = [
        sys.executable, str(RUNNER),
        "--repo-root", str(ROOT),
        "--experiment-id", eid,
        "--manifest", req("CROPCOP_MANIFEST"),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--image-root", req("CROPCOP_IMAGE_ROOT"),
        "--secondary-g1-bundle", str(g1),
        "--g2-barrier", str(g2),
        "--run-id", rid,
        "--lane-id", lane,
        "--source-git-commit", source_sha,
        "--output-dir", str(child_out),
        "--row-id-column", req("CROPCOP_ROW_ID_COLUMN"),
        "--path-column", req("CROPCOP_PATH_COLUMN"),
        "--split-column", req("CROPCOP_SPLIT_COLUMN"),
        "--label-column", req("CROPCOP_LABEL_COLUMN"),
        "--num-workers", str(operational_worker_count(os.environ)),
        "--checkpoint-every-steps", os.environ.get("CROPCOP_CHECKPOINT_EVERY_STEPS", "250"),
        "--resume-mode", resume_mode,
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



def durable_resume_mode(run_id_value: str, locator: str, access: dict) -> str:
    check = access.get("checks", {}).get(run_id_value, {})
    version = int(check.get("current_version_number") or 0)
    cp = subprocess.run(
        ["kaggle", "datasets", "files", "-d", locator],
        check=False, capture_output=True, text=True, timeout=180,
    )
    if cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip()[-800:]
        raise EnvelopeError(f"cannot inspect durable target before scientific launch {locator}: {detail}")
    has_index = "checkpoint_index.json" in cp.stdout
    if has_index:
        return "required"
    if version <= 1:
        return "never"
    raise EnvelopeError(
        f"ambiguous durable state for {run_id_value}: private target version={version} "
        "but checkpoint_index.json is absent; fresh restart is forbidden"
    )

def publish(source_sha: str, rid: str, out: Path) -> dict:
    files = [p for p in (out / "run_record.json", out / "metrics.json", out / "segments.jsonl") if p.is_file()]
    try:
        branch = publish_to_github_branch(repo_dir=ROOT, source_git_sha=source_sha, run_id=rid, files=files)
        return {"publication_status": "PASS", "publication_branch": branch}
    except Exception as exc:
        return {"publication_status": "FAIL", "publication_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    if len(source_sha) != 40:
        raise SystemExit("CROPCOP_SOURCE_GIT_COMMIT must be exact 40-char SHA")
    require_parent_batch("secondary-scientific-dual")
    budget = SessionBudget.from_environment(require_global_clock=True)

    config = load_json(envelope_path())
    errors = validate_secondary_envelope_config(config)
    if errors:
        raise SystemExit("secondary envelope invalid: " + "; ".join(errors))
    lane = config["logical_lane"]

    g1 = Path(req("CROPCOP_SECONDARY_G1_INPUT_ROOT")).resolve()
    seal, errors = validate_secondary_g1_bundle(g1)
    if errors:
        raise SystemExit("secondary G1 invalid: " + "; ".join(errors))
    if seal.get("source_git_sha") != source_sha:
        raise SystemExit("secondary G1 source differs from scientific source")

    g2 = Path(req("CROPCOP_SECONDARY_G2_BARRIER")).resolve()
    barrier = load_json(g2)
    errors = validate_secondary_g2_barrier_object(
        barrier,
        expected_source_sha=source_sha,
        expected_g1_seal_sha256=seal["secondary_g1_seal_sha256"],
    )
    if errors:
        raise SystemExit("secondary G2 invalid: " + "; ".join(errors))

    inventory = gpu_inventory()
    errors = validate_t4x2_inventory(inventory)
    if errors:
        raise SystemExit("T4X2 inventory invalid: " + "; ".join(errors))

    output_root = Path(os.environ.get("CROPCOP_OUTPUT_ROOT", "/kaggle/working/cropcop-secondary")).resolve()
    verify_clean_source(ROOT, authorized_source_sha=source_sha, output_roots=[output_root, g1, g2.parent])
    output_root.mkdir(parents=True, exist_ok=True)

    children = config["children"]
    run_ids = {c["child_id"]: run_id(c["experiment_id"], source_sha) for c in children}
    kind = req("CROPCOP_DURABLE_STORE_KIND")
    template = req("CROPCOP_DURABLE_LOCATOR_TEMPLATE")
    all_ids = list(run_ids.values())
    locators = validate_durable_locator_template(kind, template, all_ids)
    create_policy = os.environ.get("CROPCOP_SECONDARY_ALLOW_CREATE_PRIVATE_DATASETS", "0").strip()
    if create_policy not in {"0", "1"}:
        raise SystemExit("CROPCOP_SECONDARY_ALLOW_CREATE_PRIVATE_DATASETS must be 0 or 1")
    if kind == "kaggle-dataset" and create_policy == "1":
        target_env = dict(os.environ)
        target_env["CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET"] = "1"
        for locator in locators.values():
            ensure_private_target(locator, env=target_env)
    access = validate_durable_access_plan(kind, {rid: locators[rid] for rid in all_ids}, env=os.environ)
    if access.get("status") != "PASS":
        raise SystemExit("secondary durable access failed: " + "; ".join(access.get("errors", [])))
    resume_modes = {
        rid: durable_resume_mode(rid, locators[rid], access)
        for rid in all_ids
    }

    running = {}
    child_outs = {}
    for child in children:
        cid = child["child_id"]
        slot = int(child["slot"])
        rid = run_ids[cid]
        out = output_root / "children" / cid
        child_outs[cid] = out
        env = child_environment(
            base_env=os.environ,
            envelope_id=config["envelope_id"],
            physical_slot=slot,
            output_root=out,
            g2_summaries_root=output_root / "g2_unused",
            terminal_root=out / "terminal",
            workers=operational_worker_count(os.environ),
        )
        running[cid] = launch_process(
            child_id=cid,
            experiment_id=child["experiment_id"],
            run_id=rid,
            slot=slot,
            cmd=common_args(source_sha, lane, g1, g2, out, child["experiment_id"], rid, kind, locators[rid], resume_modes[rid]),
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
            if stop["value"] or budget.should_finalize(estimated_checkpoint_seconds=180, estimated_sync_seconds=240):
                grace = min(900.0, max(300.0, budget.finalization_margin_seconds * 0.75))
                gracefully_finalize_process_groups(list(running.values()), grace_seconds=grace)
                break
            time.sleep(5.0)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for child in running.values():
            if child.process.poll() is None:
                gracefully_finalize_process_groups([child], grace_seconds=120.0)
            close_child_log(child)

    results = {}
    overall = "PASS"
    continuation = False
    for child in children:
        cid = child["child_id"]
        rid = run_ids[cid]
        record_path = child_outs[cid] / "run_record.json"
        if not record_path.is_file():
            results[cid] = {"status": "FAIL", "reason": "run_record missing", "returncode": running[cid].process.poll()}
            overall = "FAIL"
            continue
        record = load_json(record_path)
        status = record.get("status")
        cont = bool(record.get("continuation_required"))
        pub = publish(source_sha, rid, child_outs[cid])
        results[cid] = {
            "experiment_id": child["experiment_id"],
            "run_id": rid,
            "execution_status": status,
            "continuation_required": cont,
            "selected_checkpoint_sha256": record.get("result_summary", {}).get("selected_checkpoint_sha256"),
            "durable_locator": record.get("durable_store", {}).get("locator"),
            **pub,
        }
        if status == "LAUNCHED" and cont:
            continuation = True
            if overall != "FAIL":
                overall = "CONTINUATION_REQUIRED"
        elif status != "PASS":
            overall = "FAIL"

    evidence = {
        "schema_version": "1.0",
        "status": overall,
        "source_git_sha": source_sha,
        "envelope_id": config["envelope_id"],
        "logical_lane": lane,
        "secondary_g1_seal_sha256": seal["secondary_g1_seal_sha256"],
        "g2_barrier_sha256": barrier["barrier_sha256"],
        "gpu_inventory": inventory,
        "durable_access": access,
        "durable_resume_modes": resume_modes,
        "children": results,
        "continuation_required": continuation,
        "scientific_semantics_changed": False,
    }
    evidence_path = output_root / "SECONDARY_ENVELOPE_EVIDENCE.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if overall in {"PASS", "CONTINUATION_REQUIRED"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
