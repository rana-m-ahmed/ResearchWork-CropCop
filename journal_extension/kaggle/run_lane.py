from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "journal_extension" / "scripts"
KAGGLE = ROOT / "journal_extension" / "kaggle"
DEPENDENCY_LOCK = ROOT / "journal_extension" / "locks" / "execution_dependency_lock.json"

REQUIRED_PATH_ENV = (
    "CROPCOP_MANIFEST",
    "CROPCOP_CLASS_MAP",
    "CROPCOP_IMAGE_ROOT",
    "CROPCOP_MNV4_PRETRAINED",
    "CROPCOP_G1_BUNDLE_DIR",
    "CROPCOP_OUTPUT_ROOT",
    "CROPCOP_G2_SUMMARIES_DIR",
    "CROPCOP_TEACHER_CHECKPOINT",
    "CROPCOP_INFRA_SMOKE_EVIDENCE",
)
COLUMN_ENV = {
    "row_id_column": "CROPCOP_ROW_ID_COLUMN",
    "path_column": "CROPCOP_PATH_COLUMN",
    "split_column": "CROPCOP_SPLIT_COLUMN",
    "class_index_column": "CROPCOP_CLASS_INDEX_COLUMN",
}
G2_IDS = ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    entrypoint = Path(cmd[1]).name if len(cmd) > 1 and str(cmd[0]) == sys.executable else Path(cmd[0]).name
    print(f"+ execute {entrypoint} [arguments redacted]")
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def _write_publication_failure(run_id: str, files: list[Path], exc: Exception) -> None:
    parent = files[0].parent if files else Path(os.environ.get("CROPCOP_OUTPUT_ROOT", "."))
    path = parent / f"{run_id}.git_publication_failure.json"
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "FAIL",
        "failure_type": type(exc).__name__,
        "scientific_run_invalidated": False,
        "evidence_chain_complete": False,
        "action": "retain Saved Version output and republish small evidence centrally",
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def publish_public_safe(run_id: str, source_sha: str, files: list[Path]) -> None:
    if not (os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        print(f"public-evidence Git publication skipped for {run_id}: credential absent")
        return
    existing = [p for p in files if p.exists()]
    if not existing:
        return
    try:
        run([
            sys.executable, str(SCRIPTS / "publish_evidence.py"),
            "--repo-dir", str(ROOT),
            "--source-git-sha", source_sha,
            "--run-id", run_id,
            *sum((["--file", str(p)] for p in existing), []),
        ])
    except Exception as exc:
        _write_publication_failure(run_id, existing, exc)
        print(f"public-evidence Git publication failed for {run_id}; scientific output remains intact")


def env_path(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def common_args(source_sha: str, lane_id: str) -> list[str]:
    workers = int(os.environ.get("CROPCOP_NUM_WORKERS_PER_CHILD", "4"))
    if workers < 0 or workers > 16:
        raise RuntimeError("CROPCOP_NUM_WORKERS_PER_CHILD must be between 0 and 16")
    args = [
        "--repo-root", str(ROOT),
        "--manifest", env_path("CROPCOP_MANIFEST"),
        "--class-map", env_path("CROPCOP_CLASS_MAP"),
        "--image-root", env_path("CROPCOP_IMAGE_ROOT"),
        "--source-git-commit", source_sha,
        "--lane-id", lane_id,
        "--num-workers", str(workers),
    ]
    for flag, env_name in COLUMN_ENV.items():
        args += ["--" + flag.replace("_", "-"), env_path(env_name)]
    return args


def resolve_run_id(item: dict, source_sha: str) -> str:
    experiment_id = item["experiment_id"]
    env_key = "CROPCOP_RUN_ID_" + "".join(ch if ch.isalnum() else "_" for ch in experiment_id).upper()
    explicit = os.environ.get(env_key, "").strip()
    if explicit:
        if not all(ch.isalnum() or ch in "._-" for ch in explicit) or len(explicit) > 160:
            raise RuntimeError(f"invalid explicit run ID in {env_key}")
        return explicit
    attempt_key = env_key + "_ATTEMPT"
    attempt = int(os.environ.get(attempt_key, "1"))
    if attempt < 1 or attempt > 99:
        raise RuntimeError(f"{attempt_key} must be between 1 and 99")
    return f"JE-{experiment_id}-{source_sha[:12]}-A{attempt:02d}"


def all_principal_run_ids(lane: dict, source_sha: str) -> list[str]:
    return [resolve_run_id(item, source_sha) for item in lane["principal"]]


def durable_args(run_id: str, *, lane: dict, source_sha: str) -> list[str]:
    from cropcop_je.persistence import validate_durable_locator_template
    kind = os.environ.get("CROPCOP_DURABLE_STORE_KIND", "").strip()
    template = os.environ.get("CROPCOP_DURABLE_LOCATOR_TEMPLATE", "").strip()
    if not kind or not template:
        raise RuntimeError("scientific/calibration execution requires durable store kind and locator template")
    validate_durable_locator_template(kind, template, all_principal_run_ids(lane, source_sha) + [
        lane["calibration"]["calibration_id"]
    ])
    locator = template.format(run_id=run_id, run_id_lower=run_id.lower())
    return ["--durable-store-kind", kind, "--durable-store-locator", locator, "--durable-required"]


def _git_fetch_branch(branch: str) -> bool:
    token = os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return False
    remote_ref = f"refs/remotes/origin/{branch}"
    with tempfile.TemporaryDirectory() as td:
        askpass = Path(td) / "askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\nimport os,sys\np=sys.argv[1] if len(sys.argv)>1 else ''\n"
            "print('x-access-token' if 'Username' in p else (os.environ.get('CROPCOP_GITHUB_TOKEN') or os.environ.get('GITHUB_TOKEN')))\n",
            encoding="utf-8",
        )
        askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
        env = dict(os.environ)
        env["GIT_ASKPASS"] = str(askpass)
        env["GIT_TERMINAL_PROMPT"] = "0"
        cp = subprocess.run(
            ["git", "fetch", "origin", f"refs/heads/{branch}:{remote_ref}"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
    return cp.returncode == 0


def collect_g2_summaries_from_evidence_branches(shared_dir: Path) -> None:
    shared_dir.mkdir(parents=True, exist_ok=True)
    for cid in G2_IDS:
        dest = shared_dir / f"{cid}.json"
        if dest.exists():
            continue
        branch = f"run-evidence/{cid}"
        if not _git_fetch_branch(branch):
            continue
        remote_ref = f"refs/remotes/origin/{branch}"
        repo_path = f"journal_extension/evidence/public/runs/{cid}/{cid}.json"
        show = subprocess.run(
            ["git", "show", f"{remote_ref}:{repo_path}"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if show.returncode == 0:
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(show.stdout, encoding="utf-8")
            os.replace(tmp, dest)


def validate_smoke(source_sha: str) -> dict:
    from cropcop_je.smoke_handoff import validate_terminal_smoke_b_evidence
    path = Path(env_path("CROPCOP_INFRA_SMOKE_EVIDENCE"))
    evidence = json.loads(path.read_text(encoding="utf-8"))
    dependency = json.loads(DEPENDENCY_LOCK.read_text(encoding="utf-8"))
    errors = validate_terminal_smoke_b_evidence(
        evidence,
        expected_source_sha=source_sha,
        expected_dependency_lock_sha256=dependency["dependency_lock_sha256"],
        require_batch=True,
    )
    if errors:
        raise RuntimeError(
            "real Kaggle terminal Smoke-B evidence is missing, stale or incomplete: "
            + "; ".join(errors)
        )
    return evidence


def validate_g1(source_sha: str, lane_id: str, output_root: Path) -> Path:
    bundle = Path(env_path("CROPCOP_G1_BUNDLE_DIR"))
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    prelaunch_report = output_root / "PRELAUNCH_VALIDATION.json"
    run([
        sys.executable, str(SCRIPTS / "validate_prelaunch.py"),
        "--repo-root", str(ROOT),
        "--authorized-source-sha", source_sha,
        "--g1-seal", str(seal_path),
        "--output-root", str(output_root),
        "--json-report", str(prelaunch_report),
    ])
    report = output_root / "G1_BARRIER.json"
    run([
        sys.executable, str(SCRIPTS / "validate_g1_barrier.py"),
        "--repo-root", str(ROOT),
        "--authorized-source-sha", source_sha,
        "--g1-bundle-dir", str(bundle),
        "--manifest", env_path("CROPCOP_MANIFEST"),
        "--class-map", env_path("CROPCOP_CLASS_MAP"),
        "--pretrained", env_path("CROPCOP_MNV4_PRETRAINED"),
        "--teacher-checkpoint", env_path("CROPCOP_TEACHER_CHECKPOINT"),
        "--teacher-factory-root", os.environ.get("CROPCOP_TEACHER_FACTORY_ROOT", str(ROOT)),
        "--infra-smoke-evidence", env_path("CROPCOP_INFRA_SMOKE_EVIDENCE"),
        "--output", str(report),
    ])
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise RuntimeError("G1 barrier did not pass")
    return bundle / "G1_MODEL_IDENTITY_SEAL.json"


def g1_paths() -> dict[str, Path]:
    bundle = Path(env_path("CROPCOP_G1_BUNDLE_DIR"))
    return {
        "bundle": bundle,
        "seal": bundle / "G1_MODEL_IDENTITY_SEAL.json",
        "evidence": bundle / "evidence",
        "private": bundle / "private",
        "factory_manifest": bundle / "evidence/TEACHER_FACTORY_BUNDLE.json",
        "class_order": bundle / "evidence/TEACHER_CLASS_ORDER_EVIDENCE.json",
    }


def calibration(lane: dict, source_sha: str) -> Path:
    cid = lane["calibration"]["calibration_id"]
    shared = Path(env_path("CROPCOP_G2_SUMMARIES_DIR"))
    shared.mkdir(parents=True, exist_ok=True)
    dest = shared / f"{cid}.json"
    if dest.exists():
        existing = json.loads(dest.read_text(encoding="utf-8"))
        if existing.get("source_git_commit") != source_sha:
            raise RuntimeError(f"stale calibration summary exists for different source SHA: {cid}")
        return dest

    paths = g1_paths()
    out = Path(env_path("CROPCOP_OUTPUT_ROOT")) / lane["lane_id"] / "calibration" / cid
    out.mkdir(parents=True, exist_ok=True)
    if lane["calibration"]["kind"] == "mnv4":
        idx = lane["seed_index"]
        cmd = [
            sys.executable, str(SCRIPTS / "calibrate.py"),
            "--config", lane["calibration"]["config"],
            "--condition", lane["calibration"]["condition"],
            "--pair-init", str(paths["private"] / f"PAIR_INIT_S{idx}.pt"),
            "--pair-init-evidence", str(paths["evidence"] / f"PAIR_INIT_S{idx}.json"),
            "--g1-seal", str(paths["seal"]),
            "--run-id", cid,
            "--output-dir", str(out),
            *common_args(source_sha, lane["lane_id"]),
        ]
        if lane["calibration"]["condition"] == "teacher":
            cmd += [
                "--teacher-checkpoint", env_path("CROPCOP_TEACHER_CHECKPOINT"),
                "--teacher-evidence", str(paths["evidence"] / "DINO_TEACHER.json"),
                "--teacher-factory", env_path("CROPCOP_TEACHER_FACTORY"),
                "--teacher-factory-manifest", str(paths["factory_manifest"]),
                "--teacher-factory-root", os.environ.get("CROPCOP_TEACHER_FACTORY_ROOT", str(ROOT)),
                "--teacher-class-order-evidence", str(paths["class_order"]),
            ]
        cmd += durable_args(cid, lane=lane, source_sha=source_sha)
        run(cmd)
    else:
        run([
            sys.executable, str(SCRIPTS / "calibrate_cnxtt.py"),
            "--pretrained", env_path("CROPCOP_CNXTT_PRETRAINED"),
            "--g1-seal", str(paths["seal"]),
            "--output-dir", str(out),
            *common_args(source_sha, lane["lane_id"]),
        ])

    src = out / "calibration_summary.json"
    if not src.exists():
        raise RuntimeError(f"calibration did not create {src}")
    shutil.copy2(src, dest)
    publish_public_safe(cid, source_sha, [dest])
    return dest


def build_g2(shared_dir: Path, output_root: Path) -> Path | None:
    collect_g2_summaries_from_evidence_branches(shared_dir)
    barrier = output_root / "G2_CALIBRATION_BARRIER.json"
    cp = run([
        sys.executable, str(SCRIPTS / "validate_g2_barrier.py"),
        "--summaries-dir", str(shared_dir),
        "--output", str(barrier),
    ], check=False)
    if cp.returncode != 0:
        return None
    return barrier


def select_principal(lane: dict) -> dict:
    explicit = os.environ.get("CROPCOP_PRINCIPAL_EXPERIMENT", "").strip()
    if not explicit:
        raise RuntimeError(
            "principal Saved Version requires explicit CROPCOP_PRINCIPAL_EXPERIMENT; "
            "calibration never falls through into science"
        )
    matches = [item for item in lane["principal"] if item["experiment_id"] == explicit]
    if len(matches) != 1:
        raise RuntimeError(f"principal experiment is not authorized for {lane['lane_id']}: {explicit}")
    return matches[0]


def phase_budget_check(item: dict, barrier: dict) -> dict:
    from cropcop_je.session import SessionBudget
    budget = SessionBudget.from_environment(require_global_clock=True)
    key = "CAL-MNV4-TEACHER" if item["condition"] == "teacher" else "CAL-MNV4-DIRECT"
    forecast = barrier["forecast"][key]
    estimated = float(forecast["estimated_run_seconds"])
    checkpoint = max(60.0, float(forecast.get("estimated_checkpoint_seconds") or 0.0))
    sync = max(120.0, float(forecast.get("estimated_durable_sync_seconds") or 0.0))
    if budget.can_start_phase(estimated, estimated_checkpoint_seconds=checkpoint, estimated_sync_seconds=sync, extra_reserve_seconds=300):
        mode = "FULL_RUN_EXPECTED_TO_FIT"
    else:
        useful_segment = min(3600.0, estimated)
        if not budget.can_start_phase(useful_segment, estimated_checkpoint_seconds=checkpoint, estimated_sync_seconds=sync, extra_reserve_seconds=300):
            raise RuntimeError("insufficient notebook-global time for a useful principal segment plus checkpoint/durable finalization")
        mode = "SEGMENTED_RUN_EXPECTED"
    return {"mode": mode, "forecast": forecast, "session": budget.snapshot()}


def run_principal(lane: dict, source_sha: str, item: dict, g2_barrier: Path) -> tuple[str, dict]:
    idx = lane["seed_index"]
    run_id = resolve_run_id(item, source_sha)
    paths = g1_paths()
    out = Path(env_path("CROPCOP_OUTPUT_ROOT")) / lane["lane_id"] / "principal" / run_id
    cmd = [
        sys.executable, str(SCRIPTS / "run_training.py"),
        "--config", item["config"],
        "--pair-init", str(paths["private"] / f"PAIR_INIT_S{idx}.pt"),
        "--pair-init-evidence", str(paths["evidence"] / f"PAIR_INIT_S{idx}.json"),
        "--g1-seal", str(paths["seal"]),
        "--g2-barrier", str(g2_barrier),
        "--run-id", run_id,
        "--output-dir", str(out),
        "--resume-mode", "auto",
        *common_args(source_sha, lane["lane_id"]),
        *durable_args(run_id, lane=lane, source_sha=source_sha),
    ]
    if item["condition"] == "teacher":
        cmd += [
            "--teacher-checkpoint", env_path("CROPCOP_TEACHER_CHECKPOINT"),
            "--teacher-evidence", str(paths["evidence"] / "DINO_TEACHER.json"),
            "--teacher-factory", env_path("CROPCOP_TEACHER_FACTORY"),
            "--teacher-factory-manifest", str(paths["factory_manifest"]),
            "--teacher-factory-root", os.environ.get("CROPCOP_TEACHER_FACTORY_ROOT", str(ROOT)),
            "--teacher-class-order-evidence", str(paths["class_order"]),
        ]
    run(cmd)
    record_path = out / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    evidence_files = [record_path]
    for candidate in (out / "metrics.json", out / "segments.jsonl"):
        if candidate.exists():
            evidence_files.append(candidate)
    publish_public_safe(run_id, source_sha, evidence_files)
    terminal_dir_raw = os.environ.get("CROPCOP_TERMINAL_EVIDENCE_DIR", "").strip()
    if terminal_dir_raw and record.get("status") == "PASS":
        terminal_dir = Path(terminal_dir_raw)
        terminal_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_path, terminal_dir / f"{item['experiment_id']}.terminal.json")
    return run_id, record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["K1", "K2", "K3"], default=os.environ.get("CROPCOP_LANE", ""))
    ap.add_argument("--phase", choices=["calibration", "principal"], default=os.environ.get("CROPCOP_EXECUTION_PHASE", ""))
    args = ap.parse_args()
    if not args.lane or not args.phase:
        raise SystemExit("select K1/K2/K3 and calibration/principal explicitly")

    missing = [x for x in REQUIRED_PATH_ENV if not os.environ.get(x)]
    missing += [x for x in COLUMN_ENV.values() if not os.environ.get(x)]
    if args.lane == "K3" and not os.environ.get("CROPCOP_CNXTT_PRETRAINED"):
        missing.append("CROPCOP_CNXTT_PRETRAINED")
    if missing:
        raise SystemExit("missing required environment variable names: " + ", ".join(sorted(set(missing))))

    source_sha = os.environ.get("CROPCOP_SOURCE_GIT_COMMIT", "").strip()
    if len(source_sha) != 40:
        raise SystemExit("CROPCOP_SOURCE_GIT_COMMIT must explicitly name the authorized immutable commit")
    lane = json.loads((KAGGLE / "lanes" / f"{args.lane}.json").read_text(encoding="utf-8"))

    from cropcop_je.session import SessionBudget
    from cropcop_je.source_state import verify_clean_source
    SessionBudget.from_environment(require_global_clock=True)
    if os.environ.get("CROPCOP_DUAL_ENVELOPE", "").strip() == "1":
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit(
                "dual-envelope child isolation failed: each child must see exactly one CUDA device"
            )
    output_roots = [
        env_path("CROPCOP_OUTPUT_ROOT"),
        env_path("CROPCOP_G1_BUNDLE_DIR"),
        env_path("CROPCOP_G2_SUMMARIES_DIR"),
    ]
    if os.environ.get("CROPCOP_TERMINAL_EVIDENCE_DIR"):
        output_roots.append(os.environ["CROPCOP_TERMINAL_EVIDENCE_DIR"])
    verify_clean_source(ROOT, authorized_source_sha=source_sha, output_roots=output_roots)
    validate_smoke(source_sha)
    g1_seal = validate_g1(source_sha, args.lane, Path(env_path("CROPCOP_OUTPUT_ROOT")) / args.lane)

    shared = Path(env_path("CROPCOP_G2_SUMMARIES_DIR"))
    output_root = Path(env_path("CROPCOP_OUTPUT_ROOT")) / args.lane

    if args.phase == "calibration":
        calibration(lane, source_sha)
        barrier = build_g2(shared, output_root)
        if barrier is None:
            print("Calibration persisted; G2 is waiting for the other authorized calibration summaries.")
            return 0
        publish_public_safe("G2-CALIBRATION-BARRIER", source_sha, [barrier])
        print("G2 barrier is PASS. This Saved Version exits; principal science requires a new explicit principal Saved Version.")
        return 0

    barrier = build_g2(shared, output_root)
    if barrier is None:
        raise SystemExit("principal launch refused: G2 barrier is not PASS")
    barrier_data = json.loads(barrier.read_text(encoding="utf-8"))
    seal_data = json.loads(Path(g1_seal).read_text(encoding="utf-8"))
    if barrier_data.get("g1_seal_sha256") != seal_data.get("g1_seal_sha256"):
        raise SystemExit("principal launch refused: G2 binds a different G1 seal")
    item = select_principal(lane)
    decision = phase_budget_check(item, barrier_data)
    print(json.dumps({"principal_phase_decision": decision["mode"], "experiment_id": item["experiment_id"]}, sort_keys=True))
    run_id, record = run_principal(lane, source_sha, item, barrier)
    if record["status"] == "LAUNCHED" and record.get("continuation_required"):
        print(f"{run_id} reached the notebook-global planned boundary; a clean Saved Version must resume the same run.")
        return 0
    if record["status"] != "PASS":
        raise SystemExit(f"principal run did not complete validly: {run_id} status={record['status']}")
    print(f"{run_id} is terminal PASS. This Saved Version exits; no second principal state is started automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
