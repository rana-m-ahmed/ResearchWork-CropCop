from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.envelope import (
    AMENDMENT_ID,
    AMENDMENT_SHA256,
    EnvelopeError,
    child_environment,
    close_child_log,
    continuation_publication_repair_set,
    continuation_skip_set,
    durable_plan,
    ensure_common_time_for_new_child,
    finalize_manifest,
    gpu_inventory,
    gpu_telemetry,
    gracefully_finalize_process_groups,
    launch_process,
    load_json,
    locate_prior_bundle,
    planned_finalization_grace_seconds,
    require_parent_batch,
    resolve_run_id,
    terminate_process_group,
    validate_prior_envelope_bundle,
    validate_disjoint_mutable_roots,
    validate_envelope_config,
    validate_manifest,
    validate_t4x2_inventory,
)
from cropcop_je.g1_barrier import from_environment as g1_barrier_from_environment, validate_g1_barrier
from cropcop_je.g1_package import mount_g1_input
from cropcop_je.g2 import build_g2_barrier, validate_calibration_summary, validate_g2_barrier_object
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.runlog import validate_run_record
from cropcop_je.science_diff import validate_science_diff
from cropcop_je.session import SessionBudget
from cropcop_je.smoke_handoff import (
    validate_terminal_dual_gpu_smoke_evidence,
    validate_terminal_smoke_b_evidence,
)
from cropcop_je.source_state import verify_clean_source

DEPENDENCY_LOCK = ROOT / "journal_extension/locks/execution_dependency_lock.json"
RUN_LANE = ROOT / "journal_extension/kaggle/run_lane.py"
ENVELOPES = ROOT / "journal_extension/kaggle/envelopes"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvelopeError(f"required envelope environment variable missing: {name}")
    return value


def _config_path() -> Path:
    explicit = os.environ.get("CROPCOP_ENVELOPE_CONFIG", "").strip()
    if explicit:
        return Path(explicit).resolve()
    phase = req("CROPCOP_EXECUTION_PHASE")
    if phase == "calibration-dual":
        return ENVELOPES / "G2_DUAL_T4.json"
    if phase == "principal-dual":
        choice = req("CROPCOP_ENVELOPE_ID")
        mapping = {
            "P1": "P1_S1_PAIR.json",
            "P2": "P2_S2_PAIR.json",
            "P3": "P3_S3_PAIR.json",
            "MGPU-P1-S1PAIR-V1": "P1_S1_PAIR.json",
            "MGPU-P2-S2PAIR-V1": "P2_S2_PAIR.json",
            "MGPU-P3-S3PAIR-V1": "P3_S3_PAIR.json",
        }
        if choice not in mapping:
            raise EnvelopeError("principal-dual CROPCOP_ENVELOPE_ID must be P1, P2 or P3")
        return ENVELOPES / mapping[choice]
    raise EnvelopeError(f"run_envelope.py does not handle phase {phase!r}")


def _smoke_preflight(source_sha: str, dependency: dict) -> dict:
    evidence = load_json(req("CROPCOP_INFRA_SMOKE_EVIDENCE"))
    errors = validate_terminal_smoke_b_evidence(
        evidence,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        require_batch=True,
    )
    if errors:
        raise EnvelopeError("canonical terminal Smoke-B preflight failed: " + "; ".join(errors))
    return evidence


def _dual_smoke_preflight(source_sha: str, dependency: dict, smoke: dict) -> dict:
    evidence = load_json(req("CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"))
    errors = validate_terminal_dual_gpu_smoke_evidence(
        evidence,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        expected_amendment_id=AMENDMENT_ID,
        expected_amendment_sha256=AMENDMENT_SHA256,
        expected_smoke_b_evidence_sha256=sha256_json(smoke),
        require_batch=True,
    )
    if errors:
        raise EnvelopeError("canonical terminal dual-GPU-smoke preflight failed: " + "; ".join(errors))
    return evidence


def _mount_and_validate_g1(source_sha: str, dependency: dict) -> tuple[Path, dict, dict]:
    input_root = req("CROPCOP_G1_INPUT_ROOT")
    mount_root = Path(
        os.environ.get("CROPCOP_G1_MOUNT_DIR", "/kaggle/working/cropcop-g1-mounted")
    ).resolve()
    if mount_root == ROOT or ROOT in mount_root.parents:
        raise EnvelopeError("G1 mount root must be outside the Git checkout")
    try:
        bundle, package = mount_g1_input(input_root, mount_root)
    except Exception as exc:
        raise EnvelopeError(f"sealed G1 package recovery failed: {type(exc).__name__}: {exc}") from exc

    os.environ["CROPCOP_G1_BUNDLE_DIR"] = str(bundle)
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    seal = load_json(seal_path)

    package_manifest = package.manifest
    if package_manifest.get("source_git_sha") != source_sha:
        raise EnvelopeError("G1 package source differs from envelope execution source")
    if package_manifest.get("dependency_lock_sha256") != dependency["dependency_lock_sha256"]:
        raise EnvelopeError("G1 package dependency lock differs from envelope")
    if package_manifest.get("g1_seal_sha256") != seal.get("g1_seal_sha256"):
        raise EnvelopeError("G1 package manifest binds a different G1 seal")

    inputs = g1_barrier_from_environment(
        ROOT,
        authorized_source_sha=source_sha,
        env=os.environ,
    )
    report = validate_g1_barrier(inputs)
    if report.get("status") != "PASS":
        raise EnvelopeError(
            "parent full G1 validation failed before GPU child launch: "
            + "; ".join(report.get("errors", []))
        )
    return seal_path, seal, package_manifest


def _central_g2_barrier(shared: Path, *, source_sha: str, g1_sha: str) -> dict:
    summaries = []
    for cid in ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT"):
        path = shared / f"{cid}.json"
        if not path.is_file():
            raise EnvelopeError(f"principal envelope requires complete central G2 summary: {cid}")
        summary = load_json(path)
        errors = validate_calibration_summary(summary)
        if errors:
            raise EnvelopeError(f"central G2 summary invalid for {cid}: " + "; ".join(errors))
        summaries.append(summary)
    barrier = build_g2_barrier(summaries)
    errors = validate_g2_barrier_object(
        barrier,
        expected_source_sha=source_sha,
        expected_g1_seal_sha256=g1_sha,
    )
    if errors:
        raise EnvelopeError("central G2 barrier preflight failed: " + "; ".join(errors))
    return barrier


def _publish(run_id: str, source_sha: str, files: list[Path]) -> str | None:
    if not (os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        raise EnvelopeError("parent Git credential is required for serialized public evidence publication")
    existing = [p for p in files if p.is_file()]
    if not existing:
        return None
    return publish_to_github_branch(
        repo_dir=ROOT,
        source_git_sha=source_sha,
        run_id=run_id,
        files=existing,
    )


def _try_publish(run_id: str, source_sha: str, files: list[Path]) -> dict:
    try:
        branch = _publish(run_id, source_sha, files)
        return {"publication_status": "PASS", "publication_branch": branch, "publication_error": None}
    except Exception as exc:
        return {
            "publication_status": "FAIL",
            "publication_branch": None,
            "publication_error": f"{type(exc).__name__}: {exc}",
        }


def _write_telemetry(path: Path) -> None:
    row = {"timestamp_utc": utc_now(), **gpu_telemetry()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def _child_paths(envelope_root: Path, child_id: str) -> dict[str, Path]:
    base = envelope_root / "children" / child_id
    return {
        "base": base,
        "output": base / "output",
        "g2": base / "g2_summaries",
        "terminal": base / "terminal",
        "log": base / "console.log",
    }


def _child_run_id(child: dict, source_sha: str) -> str:
    eid = child["experiment_id"]
    return eid if eid.startswith("CAL-") else resolve_run_id(eid, source_sha)


def _run_id_env_key(experiment_id: str) -> str:
    return "CROPCOP_RUN_ID_" + "".join(ch if ch.isalnum() else "_" for ch in experiment_id).upper()


def _validated_child_preflight(child: dict, paths: dict[str, Path], *, slot: int, envelope_id: str) -> dict:
    path = paths["terminal"] / "CHILD_PREFLIGHT.json"
    if not path.is_file():
        raise EnvelopeError(f"child isolation evidence missing: {child['child_id']}")
    row = load_json(path)
    errors = []
    if row.get("status") != "PASS":
        errors.append("status is not PASS")
    if row.get("envelope_id") != envelope_id:
        errors.append("envelope ID mismatch")
    if row.get("logical_lane") != child["lane"]:
        errors.append("logical lane mismatch")
    if row.get("requested_physical_gpu_slot") != slot:
        errors.append("physical slot mismatch")
    if row.get("cuda_visible_devices") != str(slot):
        errors.append("CUDA_VISIBLE_DEVICES mismatch")
    if row.get("observed_visible_cuda_count") != 1:
        errors.append("child did not observe exactly one CUDA device")
    if row.get("observed_visible_gpu_name") not in {"Tesla T4", "NVIDIA T4"}:
        errors.append("child visible GPU is not T4")
    if row.get("git_credentials_present") is not False:
        errors.append("child inherited Git publication credentials")
    if float(row.get("notebook_started_monotonic", -1)) != float(os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"]):
        errors.append("child notebook-global clock mismatch")
    if errors:
        raise EnvelopeError(f"child preflight invalid for {child['child_id']}: " + "; ".join(errors))
    return row


def _result_for_child(
    child: dict,
    *,
    paths: dict[str, Path],
    run_id: str,
    phase: str,
    central_g2: Path,
    source_sha: str,
) -> dict:
    if phase == "calibration-dual":
        summary_path = paths["g2"] / f"{child['experiment_id']}.json"
        if not summary_path.is_file():
            raise EnvelopeError(f"terminal calibration child did not create summary: {summary_path.name}")
        summary = load_json(summary_path)
        errors = validate_calibration_summary(summary)
        if errors:
            raise EnvelopeError(
                f"calibration child summary invalid for {child['experiment_id']}: " + "; ".join(errors)
            )
        central_g2.mkdir(parents=True, exist_ok=True)
        dest = central_g2 / summary_path.name
        if dest.exists():
            existing = load_json(dest)
            if existing != summary:
                raise EnvelopeError(f"central G2 summary collision for {child['experiment_id']}")
        else:
            tmp = dest.with_suffix(".json.tmp")
            shutil.copy2(summary_path, tmp)
            os.replace(tmp, dest)
        publication = _try_publish(child["experiment_id"], source_sha, [dest])
        return {
            "status": "PASS",
            "continuation_required": False,
            "result_relative_path": f"children/{child['child_id']}/g2_summaries/{summary_path.name}",
            **publication,
        }

    record = paths["output"] / child["lane"] / "principal" / run_id / "run_record.json"
    if not record.is_file():
        raise EnvelopeError(f"principal child did not create run record: {child['child_id']}")
    data = load_json(record)
    evidence = [record]
    for name in ("metrics.json", "segments.jsonl"):
        p = record.parent / name
        if p.is_file():
            evidence.append(p)
    publication = _try_publish(run_id, source_sha, evidence)
    status = data.get("status")
    continuation = bool(data.get("continuation_required"))
    if status == "PASS" and not continuation:
        terminal = "PASS"
    elif status == "LAUNCHED" and continuation:
        terminal = "CONTINUATION_REQUIRED"
    else:
        raise EnvelopeError(f"principal child terminal record invalid: status={status}, continuation={continuation}")
    return {
        "status": terminal,
        "continuation_required": continuation,
        "result_relative_path": f"children/{child['child_id']}/output/{child['lane']}/principal/{run_id}/run_record.json",
        **publication,
        "g1_seal_sha256": data.get("g1_seal_sha256"),
        "g2_barrier_sha256": data.get("g2_barrier_sha256"),
    }


def _prior_control_fingerprint(bundle) -> dict[str, str]:
    return {
        "manifest": sha256_file(bundle.manifest_path),
        "evidence": sha256_file(bundle.evidence_path),
        "state": sha256_file(bundle.state_path),
    }


def _copy_prior_terminal_result(
    bundle,
    *,
    child: dict,
    result: dict,
    phase: str,
    envelope_root: Path,
    central_g2: Path,
    source_sha: str,
    g1_sha: str,
    g2_sha: str | None,
) -> tuple[dict, list[Path]]:
    child_id = child["child_id"]
    rel = Path(str(result["result_relative_path"]))
    source = (bundle.root / rel).resolve()
    if bundle.root not in source.parents or not source.is_file():
        raise EnvelopeError(f"prior terminal result missing/unsafe for {child_id}")

    if phase == "calibration-dual":
        summary = load_json(source)
        errors = validate_calibration_summary(summary)
        if errors:
            raise EnvelopeError(f"prior calibration summary invalid for {child_id}: " + "; ".join(errors))
        if summary.get("source_git_commit") != source_sha:
            raise EnvelopeError(f"prior calibration summary source mismatch for {child_id}")
        if summary.get("g1_seal_sha256") != g1_sha:
            raise EnvelopeError(f"prior calibration summary G1 mismatch for {child_id}")
        dest = envelope_root / "children" / child_id / "g2_summaries" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        central_g2.mkdir(parents=True, exist_ok=True)
        central = central_g2 / source.name
        if central.exists():
            if load_json(central) != summary:
                raise EnvelopeError(f"central G2 summary collision during continuation for {child_id}")
        else:
            tmp = central.with_suffix(".json.tmp")
            shutil.copy2(source, tmp)
            os.replace(tmp, central)
        updated = dict(result)
        updated["result_relative_path"] = dest.relative_to(envelope_root).as_posix()
        return updated, [dest]

    record = load_json(source)
    validate_run_record(record)
    if record.get("run_id") != result.get("run_id", record.get("run_id")):
        raise EnvelopeError(f"prior principal run ID mismatch for {child_id}")
    if record.get("status") != "PASS" or record.get("continuation_required") is not False:
        raise EnvelopeError(f"prior principal result is not terminal PASS for {child_id}")
    if record.get("source_git_commit") != source_sha:
        raise EnvelopeError(f"prior principal source mismatch for {child_id}")
    if record.get("g1_seal_sha256") != g1_sha or record.get("g2_barrier_sha256") != g2_sha:
        raise EnvelopeError(f"prior principal G1/G2 identity mismatch for {child_id}")

    prior_dir = source.parent
    metrics = prior_dir / "metrics.json"
    segments = prior_dir / "segments.jsonl"
    if not metrics.is_file() or not segments.is_file():
        raise EnvelopeError(f"prior principal metrics/segments missing for {child_id}")
    artifact_metrics = record.get("artifact_locators", {}).get("metrics", {})
    expected_metrics_sha = artifact_metrics.get("sha256")
    if expected_metrics_sha and sha256_file(metrics) != expected_metrics_sha:
        raise EnvelopeError(f"prior principal metrics SHA mismatch for {child_id}")

    dest_dir = envelope_root / "children" / child_id / "prior_result"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in (source, metrics, segments):
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    updated = dict(result)
    updated["result_relative_path"] = copied[0].relative_to(envelope_root).as_posix()
    updated["run_id"] = record["run_id"]
    return updated, copied


def _repair_prior_child_publication(
    bundle,
    *,
    child: dict,
    prior_result: dict,
    phase: str,
    envelope_root: Path,
    central_g2: Path,
    source_sha: str,
    g1_sha: str,
    g2_sha: str | None,
    repair_publication: bool,
) -> dict:
    carried, files = _copy_prior_terminal_result(
        bundle,
        child=child,
        result=prior_result,
        phase=phase,
        envelope_root=envelope_root,
        central_g2=central_g2,
        source_sha=source_sha,
        g1_sha=g1_sha,
        g2_sha=g2_sha,
    )
    carried["status"] = "PASS"
    carried["continuation_required"] = False
    carried["carried_from_prior_envelope"] = True
    if repair_publication:
        run_id = child["experiment_id"] if phase == "calibration-dual" else carried["run_id"]
        publication = _try_publish(run_id, source_sha, files)
        carried.update(publication)
        carried["publication_repair_only"] = True
        carried["training_relaunched"] = False
    carried["evidence_chain_complete"] = carried.get("publication_status") == "PASS"
    return carried


def main() -> int:
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    if len(source_sha) != 40:
        raise EnvelopeError("CROPCOP_SOURCE_GIT_COMMIT must name the immutable 40-hex execution source")
    phase = req("CROPCOP_EXECUTION_PHASE")
    if phase not in {"calibration-dual", "principal-dual"}:
        raise EnvelopeError("run_envelope.py requires calibration-dual or principal-dual")
    run_type = require_parent_batch(phase)

    config = load_json(_config_path())
    config_errors = validate_envelope_config(config)
    if config_errors:
        raise EnvelopeError("invalid envelope config: " + "; ".join(config_errors))
    envelope_id = config["envelope_id"]

    dependency = load_json(DEPENDENCY_LOCK)
    smoke = _smoke_preflight(source_sha, dependency)
    dual_smoke = _dual_smoke_preflight(source_sha, dependency, smoke)
    _g1_path, g1, g1_package = _mount_and_validate_g1(source_sha, dependency)

    central_g2 = Path(req("CROPCOP_G2_SUMMARIES_DIR")).resolve()
    parent_output = Path(req("CROPCOP_OUTPUT_ROOT")).resolve()
    envelope_root = Path(
        os.environ.get(
            "CROPCOP_ENVELOPE_OUTPUT_ROOT",
            str(parent_output / "envelopes" / envelope_id),
        )
    ).resolve()
    if ROOT == envelope_root or ROOT in envelope_root.parents:
        raise EnvelopeError("envelope mutable root must be outside the Git checkout")
    envelope_root.mkdir(parents=True, exist_ok=True)

    verify_clean_source(
        ROOT,
        authorized_source_sha=source_sha,
        output_roots=[envelope_root, central_g2],
    )
    science = validate_science_diff(ROOT)
    if science["status"] != "PASS":
        raise EnvelopeError("science-diff sentinel failed: " + "; ".join(science["errors"]))

    inventory = gpu_inventory()
    hw_errors = validate_t4x2_inventory(inventory)
    if hw_errors:
        raise EnvelopeError("dual-T4 host preflight failed: " + "; ".join(hw_errors))

    run_ids = {child["child_id"]: _child_run_id(child, source_sha) for child in config["children"]}
    durable, durable_access = durable_plan(
        source_sha,
        current_run_ids=[run_ids[child["child_id"]] for child in config["children"]],
    )
    child_rows = []
    for child in config["children"]:
        paths = _child_paths(envelope_root, child["child_id"])
        child_rows.append(
            {
                "child_id": child["child_id"],
                "experiment_id": child["experiment_id"],
                "lane": child["lane"],
                "requested_slot": child["slot"],
                "run_id": run_ids[child["child_id"]],
                "output_root": str(paths["output"]),
                "g2_summaries_root": str(paths["g2"]),
                "terminal_root": str(paths["terminal"]),
                "durable_locator": durable[run_ids[child["child_id"]]],
            }
        )
    isolation_errors = validate_disjoint_mutable_roots(child_rows)
    if isolation_errors:
        raise EnvelopeError("mutable child isolation failed: " + "; ".join(isolation_errors))

    g2_barrier = None
    g2_sha = None
    if phase == "principal-dual":
        g2_barrier = _central_g2_barrier(
            central_g2,
            source_sha=source_sha,
            g1_sha=g1["g1_seal_sha256"],
        )
        g2_sha = g2_barrier["barrier_sha256"]

    prior_bundle = None
    prior_state = None
    prior_control_before = None
    carried_results: dict[str, dict] = {}
    input_root = os.environ.get("CROPCOP_ENVELOPE_INPUT_ROOT", "").strip()
    if input_root:
        prior_bundle = locate_prior_bundle(input_root)
        prior_state = prior_bundle.state
        prior_expected_g2 = g2_sha if g2_sha is not None else prior_bundle.state.get("g2_barrier_sha256")
        cont_errors = validate_prior_envelope_bundle(
            prior_bundle,
            envelope_id=envelope_id,
            source_sha=source_sha,
            g1_seal_sha256=g1["g1_seal_sha256"],
            g2_barrier_sha256=prior_expected_g2,
            expected_run_ids=run_ids,
        )
        if cont_errors:
            raise EnvelopeError("continuation bundle validation failed: " + "; ".join(cont_errors))
        prior_control_before = _prior_control_fingerprint(prior_bundle)

    skip = continuation_skip_set(prior_state)
    repair = continuation_publication_repair_set(prior_state)
    handled_prior = skip | repair
    if prior_bundle is not None:
        prior_results = prior_bundle.evidence.get("child_results", {})
        for child in config["children"]:
            child_id = child["child_id"]
            if child_id not in handled_prior:
                continue
            carried_results[child_id] = _repair_prior_child_publication(
                prior_bundle,
                child=child,
                prior_result=prior_results[child_id],
                phase=phase,
                envelope_root=envelope_root,
                central_g2=central_g2,
                source_sha=source_sha,
                g1_sha=g1["g1_seal_sha256"],
                g2_sha=g2_sha,
                repair_publication=(child_id in repair),
            )

    manifest = finalize_manifest(
        {
            "schema_version": "1.0",
            "envelope_id": envelope_id,
            "phase": phase,
            "amendment_id": AMENDMENT_ID,
            "amendment_sha256": AMENDMENT_SHA256,
            "source_git_sha": source_sha,
            "dependency_lock_sha256": dependency["dependency_lock_sha256"],
            "kaggle_run_type": run_type,
            "notebook_started_monotonic": float(req("CROPCOP_NOTEBOOK_STARTED_MONOTONIC")),
            "parent_gpu_inventory": inventory,
            "workers_per_child": int(os.environ.get("CROPCOP_NUM_WORKERS_PER_CHILD", "2")),
            "durable_access_preflight": durable_access,
            "no_git_child_policy": True,
            "science_diff_status": science["status"],
            "protected_surfaces_unchanged": science["protected_surfaces_unchanged"],
            "g1_seal_sha256": g1["g1_seal_sha256"],
            "g1_package_sha256": g1_package.get("package_sha256"),
            "g1_package_manifest_sha256": g1_package.get("manifest_sha256"),
            "g2_barrier_sha256": g2_sha,
            "dual_gpu_smoke_evidence_sha256": sha256_json(dual_smoke),
            "children": [
                {
                    "child_id": row["child_id"],
                    "experiment_id": row["experiment_id"],
                    "logical_lane": row["lane"],
                    "requested_physical_slot": row["requested_slot"],
                    "run_id": row["run_id"],
                    "output_root": f"children/{row['child_id']}/output",
                    "g2_summaries_root": f"children/{row['child_id']}/g2_summaries",
                    "terminal_root": f"children/{row['child_id']}/terminal",
                    "durable_locator": row["durable_locator"],
                }
                for row in child_rows
            ],
            "created_at_utc": utc_now(),
        }
    )
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise EnvelopeError("envelope manifest finalization failed: " + "; ".join(manifest_errors))
    manifest_path = envelope_root / "ENVELOPE_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)

    state_path = envelope_root / "ENVELOPE_STATE.json"
    state = {
        "schema_version": "1.0",
        "envelope_id": envelope_id,
        "phase": phase,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_git_sha": source_sha,
        "g1_seal_sha256": g1["g1_seal_sha256"],
        "g2_barrier_sha256": g2_sha,
        "state": "PREFLIGHT_PASS",
        "children": [
            {
                "child_id": child["child_id"],
                "experiment_id": child["experiment_id"],
                "run_id": run_ids[child["child_id"]],
                "status": (
                    carried_results[child["child_id"]]["status"]
                    if child["child_id"] in carried_results
                    else "PLANNED"
                ),
                "execution_status": (
                    carried_results[child["child_id"]]["status"]
                    if child["child_id"] in carried_results
                    else "PLANNED"
                ),
                "publication_status": (
                    carried_results[child["child_id"]].get("publication_status")
                    if child["child_id"] in carried_results
                    else "NOT_ATTEMPTED"
                ),
                "evidence_chain_complete": (
                    bool(carried_results[child["child_id"]].get("evidence_chain_complete"))
                    if child["child_id"] in carried_results
                    else False
                ),
            }
            for child in config["children"]
        ],
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(state_path, state)

    telemetry_path = envelope_root / "GPU_TELEMETRY.jsonl"
    workers = int(os.environ.get("CROPCOP_NUM_WORKERS_PER_CHILD", "2"))
    child_timeout = float(os.environ.get("CROPCOP_CHILD_MAX_RUNTIME_SECONDS", "0") or 0)
    running = {}
    free_slots = {0, 1}
    pending = [child for child in config["children"] if child["child_id"] not in handled_prior]
    results = dict(carried_results)
    starts = {}
    ends = {}
    global_stop = {"reason": None}
    finalization_outcomes: dict[str, dict] = {}
    finalization_started = False

    def _signal(signum, _frame):
        try:
            global_stop["reason"] = signal.Signals(signum).name
        except Exception:
            global_stop["reason"] = str(signum)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _signal)
            except (ValueError, OSError):
                pass

    last_telemetry = 0.0
    while pending or running:
        budget = SessionBudget.from_environment(require_global_clock=True)
        if budget.remaining_safe_seconds <= 300 and running and not global_stop["reason"]:
            global_stop["reason"] = "COMMON_SAFE_DEADLINE"

        if global_stop["reason"] and not finalization_started:
            pending.clear()
            grace = planned_finalization_grace_seconds(budget)
            finalization_outcomes.update(
                gracefully_finalize_process_groups(
                    list(running.values()),
                    grace_seconds=grace,
                )
            )
            finalization_started = True

        launched = not bool(global_stop["reason"])
        while launched and pending:
            launched = False
            child = pending[0]
            requested = child["slot"]
            if requested == "first_free":
                if not free_slots:
                    break
                slot = min(free_slots)
            else:
                slot = int(requested)
                if slot not in free_slots:
                    break
            ensure_common_time_for_new_child()
            pending.pop(0)
            free_slots.remove(slot)

            paths = _child_paths(envelope_root, child["child_id"])
            for key in ("output", "g2", "terminal"):
                paths[key].mkdir(parents=True, exist_ok=True)

            env = child_environment(
                base_env=dict(os.environ),
                envelope_id=envelope_id,
                physical_slot=slot,
                output_root=paths["output"],
                g2_summaries_root=paths["g2"] if phase == "calibration-dual" else central_g2,
                terminal_root=paths["terminal"],
                workers=workers,
            )
            env[_run_id_env_key(child["experiment_id"])] = run_ids[child["child_id"]]
            if phase == "principal-dual":
                env["CROPCOP_PRINCIPAL_EXPERIMENT"] = child["experiment_id"]

            cmd = [
                sys.executable,
                str(RUN_LANE),
                "--lane",
                child["lane"],
                "--phase",
                "calibration" if phase == "calibration-dual" else "principal",
            ]
            obj = launch_process(
                child_id=child["child_id"],
                experiment_id=child["experiment_id"],
                run_id=run_ids[child["child_id"]],
                slot=slot,
                cmd=cmd,
                env=env,
                cwd=ROOT,
                log_path=paths["log"],
                timeout_seconds=(child_timeout or None),
            )
            running[child["child_id"]] = obj
            starts[child["child_id"]] = obj.started_monotonic
            for row in state["children"]:
                if row["child_id"] == child["child_id"]:
                    row.update({
                        "status": "RUNNING",
                        "execution_status": "RUNNING",
                        "publication_status": "NOT_ATTEMPTED",
                        "evidence_chain_complete": False,
                        "physical_slot": slot,
                        "started_at_utc": utc_now(),
                    })
            state["state"] = "RUNNING"
            state["updated_at_utc"] = utc_now()
            atomic_write_json(state_path, state)
            launched = True

        now = time.monotonic()
        if now - last_telemetry >= 5.0:
            _write_telemetry(telemetry_path)
            last_telemetry = now

        any_terminal = False
        for child_id, obj in list(running.items()):
            if obj.timeout_seconds and now - obj.started_monotonic > obj.timeout_seconds and obj.process.poll() is None:
                termination_started = time.monotonic()
                mode = terminate_process_group(obj)
                finalization_outcomes[child_id] = {
                    "termination_mode": mode,
                    "termination_duration_seconds": max(0.0, time.monotonic() - termination_started),
                    "returncode": obj.process.poll(),
                    "reason": "CHILD_RUNTIME_TIMEOUT",
                }
            rc = obj.process.poll()
            if rc is None:
                continue
            any_terminal = True
            ends[child_id] = time.monotonic()
            free_slots.add(obj.slot)
            close_child_log(obj)
            del running[child_id]
            child = next(x for x in config["children"] if x["child_id"] == child_id)
            paths = _child_paths(envelope_root, child_id)
            try:
                preflight = _validated_child_preflight(child, paths, slot=obj.slot, envelope_id=envelope_id)
                result = _result_for_child(
                    child,
                    paths=paths,
                    run_id=obj.run_id,
                    phase=phase,
                    central_g2=central_g2,
                    source_sha=source_sha,
                )
                result["child_preflight"] = preflight
                result["process_returncode"] = rc
            except Exception as exc:
                result = {
                    "status": "CONTINUATION_REQUIRED" if global_stop["reason"] else "FAIL_TECHNICAL",
                    "continuation_required": bool(global_stop["reason"]),
                    "returncode": rc,
                    "error": f"{type(exc).__name__}: {exc}",
                    "publication_status": "NOT_ATTEMPTED",
                    "evidence_chain_complete": False,
                }
            result.update(
                {
                    "physical_slot": obj.slot,
                    "physical_gpu_uuid": inventory[obj.slot]["uuid"],
                    "started_monotonic": starts[child_id],
                    "ended_monotonic": ends[child_id],
                    "duration_seconds": max(0.0, ends[child_id] - starts[child_id]),
                    "child_visible_cuda_count_required": 1,
                    "git_credentials_present_in_child": False,
                    "console_log": f"children/{child_id}/console.log",
                    "termination": finalization_outcomes.get(child_id),
                    "evidence_chain_complete": (
                        result.get("status") == "PASS"
                        and result.get("publication_status") == "PASS"
                    ),
                }
            )
            results[child_id] = result
            for row in state["children"]:
                if row["child_id"] == child_id:
                    row.update(
                        {
                            "status": result["status"],
                            "execution_status": result["status"],
                            "publication_status": result.get("publication_status", "NOT_ATTEMPTED"),
                            "evidence_chain_complete": bool(result.get("evidence_chain_complete")),
                            "physical_slot": obj.slot,
                            "ended_at_utc": utc_now(),
                            "continuation_required": result.get("continuation_required", False),
                            "termination": finalization_outcomes.get(child_id),
                        }
                    )
            state["state"] = "PARTIAL_TERMINAL" if pending or running else "RUNNING"
            state["updated_at_utc"] = utc_now()
            atomic_write_json(state_path, state)

        if not any_terminal:
            time.sleep(0.5)

    for obj in list(running.values()):
        termination_started = time.monotonic()
        mode = terminate_process_group(obj)
        finalization_outcomes[obj.child_id] = {
            "termination_mode": mode,
            "termination_duration_seconds": max(0.0, time.monotonic() - termination_started),
            "returncode": obj.process.poll(),
            "reason": "POST_LOOP_EMERGENCY_CLEANUP",
        }
        close_child_log(obj)

    if any(row.get("execution_status", row.get("status")) == "FAIL_TECHNICAL" for row in state["children"]):
        envelope_status = "FAIL_TECHNICAL"
    elif any(row.get("execution_status", row.get("status")) == "CONTINUATION_REQUIRED" for row in state["children"]):
        envelope_status = "CONTINUATION_REQUIRED"
    elif all(row.get("execution_status", row.get("status")) == "PASS" for row in state["children"]):
        if all(row.get("evidence_chain_complete") is True for row in state["children"]):
            envelope_status = "PASS"
        else:
            envelope_status = "FAIL_TECHNICAL"
    else:
        envelope_status = "FAIL_TECHNICAL"

    final_g2 = None
    final_g2_branch = None
    if phase == "calibration-dual" and envelope_status == "PASS":
        summaries = [load_json(central_g2 / f"{cid}.json") for cid in ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT")]
        final_g2 = build_g2_barrier(summaries)
        g2_errors = validate_g2_barrier_object(
            final_g2,
            expected_source_sha=source_sha,
            expected_g1_seal_sha256=g1["g1_seal_sha256"],
        )
        if g2_errors:
            envelope_status = "FAIL_TECHNICAL"
            final_g2["status"] = "FAIL"
            final_g2.setdefault("errors", []).extend(g2_errors)
        g2_path = envelope_root / "G2_CALIBRATION_BARRIER.json"
        atomic_write_json(g2_path, final_g2)
        if envelope_status == "PASS":
            g2_publication = _try_publish("G2-CALIBRATION-BARRIER", source_sha, [g2_path])
            final_g2_branch = g2_publication.get("publication_branch")
            if g2_publication.get("publication_status") != "PASS":
                envelope_status = "FAIL_TECHNICAL"

        manifest["g2_barrier_sha256"] = final_g2.get("barrier_sha256")
        manifest = finalize_manifest(manifest)
        atomic_write_json(manifest_path, manifest)

    start_values = list(starts.values())
    end_values = list(ends.values())
    overlap = 0.0
    if len(start_values) >= 2 and len(end_values) >= 2:
        ids = list(starts)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if a in ends and b in ends:
                    overlap = max(overlap, max(0.0, min(ends[a], ends[b]) - max(starts[a], starts[b])))

    evidence = {
        "schema_version": "1.0",
        "status": envelope_status,
        "envelope_id": envelope_id,
        "phase": phase,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_git_sha": source_sha,
        "dependency_lock_sha256": dependency["dependency_lock_sha256"],
        "kaggle_run_type": run_type,
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_gpu_inventory": inventory,
        "child_results": results,
        "overlap_duration_seconds": overlap,
        "no_git_child_policy": True,
        "parent_publication_serialized": True,
        "durable_access_preflight": durable_access,
        "science_diff_status": science["status"],
        "protected_surfaces_unchanged": science["protected_surfaces_unchanged"],
        "g1_seal_sha256": g1["g1_seal_sha256"],
        "g2_barrier_sha256": (final_g2 or g2_barrier or {}).get("barrier_sha256"),
        "dual_gpu_smoke_evidence_sha256": sha256_json(dual_smoke),
        "g2_publication_branch": final_g2_branch,
        "host_global_stop_reason": global_stop["reason"],
        "finalization_outcomes": finalization_outcomes,
        "created_at_utc": utc_now(),
        "scientific_configuration_changed": False,
    }
    evidence_path = envelope_root / "ENVELOPE_EVIDENCE.json"
    atomic_write_json(evidence_path, evidence)

    state["state"] = envelope_status
    state["g2_barrier_sha256"] = evidence["g2_barrier_sha256"]
    state["updated_at_utc"] = utc_now()
    atomic_write_json(state_path, state)

    if prior_bundle is not None and prior_control_before != _prior_control_fingerprint(prior_bundle):
        raise EnvelopeError("attached prior envelope control files changed during continuation")

    envelope_branch = _publish(envelope_id, source_sha, [manifest_path, evidence_path, state_path])
    evidence["envelope_publication_branch"] = envelope_branch
    atomic_write_json(evidence_path, evidence)
    envelope_branch2 = _publish(envelope_id, source_sha, [manifest_path, evidence_path, state_path])
    if envelope_branch2 != envelope_branch:
        raise EnvelopeError("serialized envelope publication branch changed unexpectedly")

    print(json.dumps({
        "status": envelope_status,
        "envelope_id": envelope_id,
        "source_git_sha": source_sha,
        "g1_seal_sha256": g1["g1_seal_sha256"],
        "g2_barrier_sha256": evidence["g2_barrier_sha256"],
        "child_status": {row["child_id"]: row["status"] for row in state["children"]},
        "overlap_duration_seconds": overlap,
        "scientific_result_interpreted_by_parent": False,
    }, indent=2, sort_keys=True))
    return 0 if envelope_status in {"PASS", "CONTINUATION_REQUIRED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
