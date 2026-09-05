from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_json
from .hashing import sha256_json
from .persistence import validate_durable_access_plan, validate_durable_locator_template
from .session import SessionBudget
from .smoke_handoff import require_qualifying_kaggle_batch

AMENDMENT_ID = "EAAI-JE-MGPU-A1"
AMENDMENT_SHA256 = "3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6"
T4_NAMES = {"Tesla T4", "NVIDIA T4"}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
PRINCIPAL_MAP = {
    "MGPU-P1-S1PAIR-V1": {
        "R04-MNV4-DIRECT-S1": 0,
        "R05-MNV4-TEACHER-S1": 1,
    },
    "MGPU-P2-S2PAIR-V1": {
        "R04-MNV4-DIRECT-S2": 0,
        "R05-MNV4-TEACHER-S2": 1,
    },
    "MGPU-P3-S3PAIR-V1": {
        "R04-MNV4-DIRECT-S3": 0,
        "R05-MNV4-TEACHER-S3": 1,
    },
}
CALIBRATION_IDS = ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT")


class EnvelopeError(RuntimeError):
    pass


@dataclass(frozen=True)
class PriorEnvelopeBundle:
    root: Path
    manifest_path: Path
    evidence_path: Path
    state_path: Path
    manifest: dict[str, Any]
    evidence: dict[str, Any]
    state: dict[str, Any]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_self_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("manifest_sha256", None)
    return sha256_json(clean)


def finalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = manifest_self_hash(result)
    return result


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("manifest_sha256") != manifest_self_hash(payload):
        errors.append("envelope manifest self-hash mismatch")
    if payload.get("amendment_id") != AMENDMENT_ID:
        errors.append("envelope manifest amendment ID mismatch")
    if payload.get("amendment_sha256") != AMENDMENT_SHA256:
        errors.append("envelope manifest amendment SHA mismatch")
    return errors


def validate_envelope_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "1.0":
        errors.append("unsupported envelope config schema")
    envelope_id = str(config.get("envelope_id", ""))
    phase = config.get("phase")
    if phase not in {"calibration-dual", "principal-dual"}:
        errors.append("envelope phase must be calibration-dual or principal-dual")
    if config.get("hardware_profile") != "T4X2":
        errors.append("envelope hardware profile must be T4X2")
    children = config.get("children")
    if not isinstance(children, list) or not children:
        return errors + ["envelope children missing"]

    child_ids = [str(c.get("child_id", "")) for c in children]
    experiments = [str(c.get("experiment_id", "")) for c in children]
    if any(not x for x in child_ids) or len(set(child_ids)) != len(child_ids):
        errors.append("envelope child IDs missing or non-unique")
    if any(not x for x in experiments) or len(set(experiments)) != len(experiments):
        errors.append("envelope experiment IDs missing or non-unique")

    for child in children:
        if child.get("lane") not in {"K1", "K2", "K3"}:
            errors.append(f"invalid logical lane for {child.get('child_id')}")
        if child.get("slot") not in {0, 1, "first_free"}:
            errors.append(f"invalid physical slot for {child.get('child_id')}")

    if phase == "calibration-dual":
        if envelope_id != "MGPU-G2-T4X2-V1":
            errors.append("calibration-dual envelope ID mismatch")
        if tuple(experiments) != CALIBRATION_IDS:
            errors.append("G2 calibration queue/order changed")
        if [children[0].get("slot"), children[1].get("slot"), children[2].get("slot")] != [0, 1, "first_free"]:
            errors.append("G2 slot policy must be GPU0, GPU1, first_free")
    else:
        expected = PRINCIPAL_MAP.get(envelope_id)
        if expected is None:
            errors.append("principal envelope ID is not predeclared")
        else:
            actual = {str(c.get("experiment_id")): c.get("slot") for c in children}
            if actual != expected:
                errors.append("principal envelope experiment/slot mapping changed")
            lanes = {str(c.get("lane")) for c in children}
            expected_lane = {"MGPU-P1-S1PAIR-V1": "K1", "MGPU-P2-S2PAIR-V1": "K2", "MGPU-P3-S3PAIR-V1": "K3"}[envelope_id]
            if lanes != {expected_lane}:
                errors.append("principal pair logical lane mismatch")
    return errors


def _nvidia_query(fields: str) -> list[list[str]]:
    cp = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows: list[list[str]] = []
    for line in cp.stdout.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def gpu_inventory() -> list[dict[str, Any]]:
    rows = _nvidia_query("index,uuid,name,memory.total")
    result = []
    for row in rows:
        if len(row) < 4:
            raise EnvelopeError(f"unexpected nvidia-smi inventory row: {row}")
        result.append(
            {
                "index": int(row[0]),
                "uuid": row[1],
                "name": row[2],
                "memory_total_mib": int(float(row[3])),
            }
        )
    return result


def validate_t4x2_inventory(inventory: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(inventory) != 2:
        errors.append(f"T4X2 profile requires exactly 2 physical GPUs; observed {len(inventory)}")
        return errors
    if [row.get("index") for row in inventory] != [0, 1]:
        errors.append("T4X2 profile requires physical GPU indexes 0 and 1")
    names = [str(row.get("name", "")) for row in inventory]
    if any(name not in T4_NAMES for name in names):
        errors.append(f"T4X2 profile requires homogeneous Tesla/NVIDIA T4 GPUs; observed {names}")
    uuids = [str(row.get("uuid", "")) for row in inventory]
    if any(not x for x in uuids) or len(set(uuids)) != 2:
        errors.append("T4X2 profile requires two distinct physical GPU UUIDs")
    return errors


def gpu_telemetry() -> dict[str, Any]:
    fields = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    output: dict[str, Any] = {"gpus": [], "compute_processes": []}
    try:
        rows = _nvidia_query(fields)
        for row in rows:
            output["gpus"].append(
                {
                    "index": int(row[0]),
                    "uuid": row[1],
                    "name": row[2],
                    "gpu_utilization_percent": row[3],
                    "memory_used_mib": row[4],
                    "memory_total_mib": row[5],
                    "temperature_c": row[6],
                    "power_w": row[7] if len(row) > 7 else None,
                }
            )
    except Exception as exc:
        output["gpu_telemetry_error"] = f"{type(exc).__name__}: {exc}"

    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if cp.returncode == 0:
            for line in cp.stdout.splitlines():
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 4:
                    output["compute_processes"].append(
                        {
                            "pid": int(parts[0]),
                            "gpu_uuid": parts[1],
                            "process_name": parts[2],
                            "used_memory_mib": parts[3],
                        }
                    )
        else:
            output["compute_process_query_error"] = (cp.stderr or cp.stdout or "").strip()[-500:]
    except Exception as exc:
        output["compute_process_query_error"] = f"{type(exc).__name__}: {exc}"
    return output


def resolve_run_id(experiment_id: str, source_sha: str, env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    key = "CROPCOP_RUN_ID_" + "".join(ch if ch.isalnum() else "_" for ch in experiment_id).upper()
    explicit = str(env.get(key, "")).strip()
    if explicit:
        if len(explicit) > 160 or not all(ch.isalnum() or ch in "._-" for ch in explicit):
            raise EnvelopeError(f"invalid explicit run ID in {key}")
        return explicit
    attempt = int(str(env.get(key + "_ATTEMPT", "1")))
    if not 1 <= attempt <= 99:
        raise EnvelopeError(f"{key}_ATTEMPT must be 1..99")
    return f"JE-{experiment_id}-{source_sha[:12]}-A{attempt:02d}"


def all_reserved_run_ids(source_sha: str, env: dict[str, str] | None = None) -> list[str]:
    principal = [
        f"R04-MNV4-DIRECT-S{i}" for i in (1, 2, 3)
    ] + [
        f"R05-MNV4-TEACHER-S{i}" for i in (1, 2, 3)
    ]
    return [*CALIBRATION_IDS, *(resolve_run_id(eid, source_sha, env) for eid in principal)]


def durable_plan(source_sha: str, env: dict[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(os.environ if env is None else env)
    kind = str(env.get("CROPCOP_DURABLE_STORE_KIND", "")).strip()
    template = str(env.get("CROPCOP_DURABLE_LOCATOR_TEMPLATE", "")).strip()
    if not kind or not template:
        raise EnvelopeError("G2/principal envelope requires durable store kind and run-specific locator template")
    if kind == "kaggle-dataset":
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            if not str(env.get(key, "")).strip():
                raise EnvelopeError(f"production Kaggle durability requires {key}")
    resolved = validate_durable_locator_template(kind, template, all_reserved_run_ids(source_sha, env))
    access = validate_durable_access_plan(kind, resolved, env=env)
    if access.get("status") != "PASS":
        raise EnvelopeError("production durable access preflight failed: " + "; ".join(access.get("errors", [])))
    return resolved, access


def child_environment(
    *,
    base_env: dict[str, str],
    envelope_id: str,
    physical_slot: int,
    output_root: str | Path,
    g2_summaries_root: str | Path,
    terminal_root: str | Path,
    workers: int = 2,
) -> dict[str, str]:
    if physical_slot not in {0, 1}:
        raise EnvelopeError(f"invalid child physical slot: {physical_slot}")
    if workers < 0 or workers > 16:
        raise EnvelopeError("workers must be between 0 and 16")
    env = dict(base_env)
    env.pop("CROPCOP_GITHUB_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    env.update(THREAD_ENV)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_slot),
            "CROPCOP_DUAL_ENVELOPE": "1",
            "CROPCOP_ENVELOPE_ID": envelope_id,
            "CROPCOP_PHYSICAL_GPU_SLOT": str(physical_slot),
            "CROPCOP_NUM_WORKERS_PER_CHILD": str(workers),
            "CROPCOP_OUTPUT_ROOT": str(Path(output_root).resolve()),
            "CROPCOP_G2_SUMMARIES_DIR": str(Path(g2_summaries_root).resolve()),
            "CROPCOP_TERMINAL_EVIDENCE_DIR": str(Path(terminal_root).resolve()),
        }
    )
    return env


def validate_disjoint_mutable_roots(children: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    fields = ("output_root", "g2_summaries_root", "terminal_root")
    resolved: list[tuple[str, str, Path]] = []
    for child in children:
        cid = str(child["child_id"])
        for field in fields:
            value = child.get(field)
            if not value:
                errors.append(f"{cid}: missing mutable root {field}")
                continue
            resolved.append((cid, field, Path(value).resolve()))
    for i, (cid_a, field_a, a) in enumerate(resolved):
        for cid_b, field_b, b in resolved[i + 1 :]:
            if cid_a == cid_b:
                continue
            if a == b or a in b.parents or b in a.parents:
                errors.append(
                    f"child mutable-root collision: {cid_a}:{field_a}={a} vs {cid_b}:{field_b}={b}"
                )
    return errors


def locate_prior_bundle(input_root: str | Path) -> PriorEnvelopeBundle:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise EnvelopeError(f"CROPCOP_ENVELOPE_INPUT_ROOT is not a directory: {root}")
    names = ("ENVELOPE_MANIFEST.json", "ENVELOPE_EVIDENCE.json", "ENVELOPE_STATE.json")
    found: dict[str, list[Path]] = {
        name: sorted(path.resolve() for path in root.rglob(name) if path.is_file())
        for name in names
    }
    for name, matches in found.items():
        if len(matches) != 1:
            raise EnvelopeError(
                f"continuation input must contain exactly one {name}; found {len(matches)}"
            )
    parents = {found[name][0].parent for name in names}
    if len(parents) != 1:
        raise EnvelopeError("continuation manifest/evidence/state must come from the same envelope directory")
    envelope_root = next(iter(parents))
    return PriorEnvelopeBundle(
        root=envelope_root,
        manifest_path=found["ENVELOPE_MANIFEST.json"][0],
        evidence_path=found["ENVELOPE_EVIDENCE.json"][0],
        state_path=found["ENVELOPE_STATE.json"][0],
        manifest=load_json(found["ENVELOPE_MANIFEST.json"][0]),
        evidence=load_json(found["ENVELOPE_EVIDENCE.json"][0]),
        state=load_json(found["ENVELOPE_STATE.json"][0]),
    )


def validate_continuation_state(
    state: dict[str, Any],
    *,
    envelope_id: str,
    source_sha: str,
    g1_seal_sha256: str | None,
    g2_barrier_sha256: str | None,
    expected_run_ids: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    expected = {
        "envelope_id": envelope_id,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_git_sha": source_sha,
    }
    for field, value in expected.items():
        if state.get(field) != value:
            errors.append(f"continuation {field} mismatch")
    if state.get("g1_seal_sha256") != g1_seal_sha256:
        errors.append("continuation G1 identity mismatch")
    if state.get("g2_barrier_sha256") != g2_barrier_sha256:
        errors.append("continuation G2 identity mismatch")
    observed = {
        str(row.get("child_id")): str(row.get("run_id"))
        for row in state.get("children", [])
    }
    if observed != expected_run_ids:
        errors.append("continuation child run-ID mapping mismatch")
    return errors


def validate_prior_envelope_bundle(
    bundle: PriorEnvelopeBundle,
    *,
    envelope_id: str,
    source_sha: str,
    g1_seal_sha256: str | None,
    g2_barrier_sha256: str | None,
    expected_run_ids: dict[str, str],
) -> list[str]:
    errors = validate_manifest(bundle.manifest)
    common = {
        "envelope_id": envelope_id,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_git_sha": source_sha,
        "g1_seal_sha256": g1_seal_sha256,
        "g2_barrier_sha256": g2_barrier_sha256,
    }
    for label, obj in (
        ("manifest", bundle.manifest),
        ("evidence", bundle.evidence),
        ("state", bundle.state),
    ):
        for field, expected in common.items():
            if obj.get(field) != expected:
                errors.append(f"prior {label} {field} mismatch")
    if bundle.evidence.get("manifest_sha256") != bundle.manifest.get("manifest_sha256"):
        errors.append("prior evidence does not bind prior manifest SHA")
    errors.extend(
        validate_continuation_state(
            bundle.state,
            envelope_id=envelope_id,
            source_sha=source_sha,
            g1_seal_sha256=g1_seal_sha256,
            g2_barrier_sha256=g2_barrier_sha256,
            expected_run_ids=expected_run_ids,
        )
    )

    manifest_runs = {
        str(row.get("child_id")): str(row.get("run_id"))
        for row in bundle.manifest.get("children", [])
    }
    if manifest_runs != expected_run_ids:
        errors.append("prior manifest child run-ID mapping mismatch")
    results = bundle.evidence.get("child_results")
    if not isinstance(results, dict):
        errors.append("prior evidence child_results missing/invalid")
        results = {}
    state_rows = {
        str(row.get("child_id")): row
        for row in bundle.state.get("children", [])
    }
    if set(state_rows) != set(expected_run_ids):
        errors.append("prior state child set mismatch")

    for child_id, run_id in expected_run_ids.items():
        state_row = state_rows.get(child_id, {})
        result = results.get(child_id)
        if not isinstance(result, dict):
            errors.append(f"prior evidence result missing for {child_id}")
            continue
        execution_status = state_row.get("execution_status", state_row.get("status"))
        if result.get("status") != execution_status:
            errors.append(f"prior child execution status mismatch for {child_id}")
        publication_status = state_row.get("publication_status")
        if result.get("publication_status") != publication_status:
            errors.append(f"prior child publication status mismatch for {child_id}")
        complete = bool(state_row.get("evidence_chain_complete"))
        expected_complete = (
            execution_status == "PASS"
            and publication_status == "PASS"
            and result.get("publication_status") == "PASS"
        )
        if complete != expected_complete:
            errors.append(f"prior child evidence-completeness mismatch for {child_id}")

        if execution_status == "PASS":
            rel_text = str(result.get("result_relative_path", ""))
            rel = Path(rel_text)
            if not rel_text or rel.is_absolute() or ".." in rel.parts:
                errors.append(f"prior child terminal result path invalid for {child_id}")
                continue
            artifact = (bundle.root / rel).resolve()
            if bundle.root not in artifact.parents:
                errors.append(f"prior child terminal result escapes envelope root for {child_id}")
            elif not artifact.is_file():
                errors.append(f"prior child terminal result missing for {child_id}: {rel_text}")

    return errors


def continuation_skip_set(state: dict[str, Any] | None) -> set[str]:
    if not state:
        return set()
    return {
        str(row.get("child_id"))
        for row in state.get("children", [])
        if row.get("execution_status", row.get("status")) == "PASS"
        and row.get("publication_status") == "PASS"
        and row.get("evidence_chain_complete") is True
    }


def continuation_publication_repair_set(state: dict[str, Any] | None) -> set[str]:
    if not state:
        return set()
    return {
        str(row.get("child_id"))
        for row in state.get("children", [])
        if row.get("execution_status", row.get("status")) == "PASS"
        and row.get("publication_status") == "FAIL"
        and row.get("evidence_chain_complete") is False
    }


def ensure_common_time_for_new_child(*, minimum_useful_seconds: float = 900.0) -> dict[str, Any]:
    budget = SessionBudget.from_environment(require_global_clock=True)
    if not budget.can_start_phase(
        minimum_useful_seconds,
        estimated_checkpoint_seconds=120,
        estimated_sync_seconds=180,
        extra_reserve_seconds=300,
    ):
        raise EnvelopeError("insufficient common notebook-safe time to launch another child")
    return budget.snapshot()


def require_parent_batch(phase: str) -> str:
    return require_qualifying_kaggle_batch(context=f"MGPU envelope phase {phase}")


@dataclass
class RunningChild:
    child_id: str
    experiment_id: str
    run_id: str
    slot: int
    process: subprocess.Popen
    log_handle: Any
    log_path: Path
    started_monotonic: float
    started_utc_epoch: float
    timeout_seconds: float | None = None


def launch_process(
    *,
    child_id: str,
    experiment_id: str,
    run_id: str,
    slot: int,
    cmd: list[str],
    env: dict[str, str],
    cwd: str | Path,
    log_path: str | Path,
    timeout_seconds: float | None = None,
) -> RunningChild:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        cmd,
        cwd=Path(cwd),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    now_mono = time.monotonic()
    return RunningChild(
        child_id=child_id,
        experiment_id=experiment_id,
        run_id=run_id,
        slot=slot,
        process=process,
        log_handle=handle,
        log_path=log,
        started_monotonic=now_mono,
        started_utc_epoch=time.time(),
        timeout_seconds=timeout_seconds,
    )


def terminate_process_group(child: RunningChild, *, grace_seconds: float = 30.0) -> str:
    """Emergency/hung-child termination. Planned finalization uses the larger concurrent helper."""
    if child.process.poll() is not None:
        return "already_terminal"
    try:
        os.killpg(child.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_terminal"
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while child.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if child.process.poll() is None:
        try:
            os.killpg(child.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return "sigkill_after_emergency_grace"
    return "sigterm_emergency"


def planned_finalization_grace_seconds(
    budget: SessionBudget,
    *,
    estimated_checkpoint_seconds: float = 120.0,
    estimated_sync_seconds: float = 180.0,
) -> float:
    desired = max(
        300.0,
        2.0 * (max(0.0, estimated_checkpoint_seconds) + max(0.0, estimated_sync_seconds)) + 60.0,
    )
    margin_cap = max(60.0, 0.8 * budget.finalization_margin_seconds)
    hard_cap = max(1.0, budget.remaining_hard_seconds - 30.0)
    return max(1.0, min(desired, margin_cap, hard_cap))


def gracefully_finalize_process_groups(
    children: list[RunningChild],
    *,
    grace_seconds: float,
    poll_seconds: float = 0.2,
) -> dict[str, dict[str, Any]]:
    """SIGTERM all children together, allow bounded checkpoint/sync finalization, then kill only hung groups."""
    started = time.monotonic()
    outcomes: dict[str, dict[str, Any]] = {}
    active: list[RunningChild] = []
    for child in children:
        if child.process.poll() is not None:
            outcomes[child.child_id] = {
                "termination_mode": "already_terminal",
                "termination_duration_seconds": 0.0,
                "returncode": child.process.poll(),
            }
            continue
        try:
            os.killpg(child.process.pid, signal.SIGTERM)
            active.append(child)
        except ProcessLookupError:
            outcomes[child.child_id] = {
                "termination_mode": "already_terminal",
                "termination_duration_seconds": max(0.0, time.monotonic() - started),
                "returncode": child.process.poll(),
            }

    deadline = started + max(0.0, grace_seconds)
    while active and time.monotonic() < deadline:
        active = [child for child in active if child.process.poll() is None]
        if active:
            time.sleep(max(0.01, poll_seconds))

    for child in children:
        if child.child_id in outcomes:
            continue
        if child.process.poll() is None:
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.process.wait(timeout=5.0)
            except Exception:
                pass
            mode = "sigkill_after_finalization_grace"
        else:
            mode = "graceful_sigterm"
        outcomes[child.child_id] = {
            "termination_mode": mode,
            "termination_duration_seconds": max(0.0, time.monotonic() - started),
            "returncode": child.process.poll(),
        }
    return outcomes


def close_child_log(child: RunningChild) -> None:
    try:
        child.log_handle.flush()
    finally:
        child.log_handle.close()


def write_state(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)
