from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "journal_extension" / "scripts"
KAGGLE = ROOT / "journal_extension" / "kaggle"

REQUIRED_PATH_ENV = (
    "CROPCOP_MANIFEST",
    "CROPCOP_CLASS_MAP",
    "CROPCOP_IMAGE_ROOT",
    "CROPCOP_MNV4_PRETRAINED",
    "CROPCOP_PAIR_INIT_DIR",
    "CROPCOP_G1_EVIDENCE_DIR",
    "CROPCOP_OUTPUT_ROOT",
    "CROPCOP_G2_SUMMARIES_DIR",
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
        "action": "retain Kaggle Saved Version output and republish small evidence centrally",
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
        print(f"public-evidence Git publication failed for {run_id}; scientific output remains intact and failure marker was persisted")


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def env_path(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def common_args(source_sha: str, lane_id: str) -> list[str]:
    args = [
        "--repo-root", str(ROOT),
        "--manifest", env_path("CROPCOP_MANIFEST"),
        "--class-map", env_path("CROPCOP_CLASS_MAP"),
        "--image-root", env_path("CROPCOP_IMAGE_ROOT"),
        "--source-git-commit", source_sha,
        "--lane-id", lane_id,
    ]
    for flag, env_name in COLUMN_ENV.items():
        args += ["--" + flag.replace("_", "-"), env_path(env_name)]
    return args


def durable_args(run_id: str) -> list[str]:
    kind = os.environ.get("CROPCOP_DURABLE_STORE_KIND", "").strip()
    template = os.environ.get("CROPCOP_DURABLE_LOCATOR_TEMPLATE", "").strip()
    if not kind or not template:
        raise RuntimeError("scientific lane requires CROPCOP_DURABLE_STORE_KIND and CROPCOP_DURABLE_LOCATOR_TEMPLATE")
    locator = template.format(run_id=run_id, run_id_lower=run_id.lower())
    return ["--durable-store-kind", kind, "--durable-store-locator", locator, "--durable-required"]


def ensure_g1(lane: dict) -> None:
    evidence = Path(env_path("CROPCOP_G1_EVIDENCE_DIR"))
    pair_dir = Path(env_path("CROPCOP_PAIR_INIT_DIR"))
    evidence.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    idx = int(lane["seed_index"])
    pair_ev = evidence / f"PAIR_INIT_S{idx}.json"
    pair_bin = pair_dir / f"PAIR_INIT_S{idx}.pt"
    if not pair_ev.exists() or not pair_bin.exists():
        run([
            sys.executable, str(SCRIPTS / "prepare_pair_init.py"),
            "--seed-index", str(idx),
            "--pretrained", env_path("CROPCOP_MNV4_PRETRAINED"),
            "--pretrained-evidence", str(evidence / "MNV4_PRETRAINED.json"),
            "--output", str(pair_bin),
            "--evidence", str(pair_ev),
        ])
    needs_teacher = lane["calibration"].get("condition") == "teacher" or any(x["condition"] == "teacher" for x in lane["principal"])
    if needs_teacher:
        teacher_ev = evidence / "DINO_TEACHER.json"
        run([
            sys.executable, str(SCRIPTS / "verify_artifact.py"),
            "--artifact", env_path("CROPCOP_TEACHER_CHECKPOINT"),
            "--kind", "teacher",
            "--evidence", str(teacher_ev),
            "--expected-sha256", "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79",
            "--class-map-sha256", "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2",
        ])


def calibration(lane: dict, source_sha: str) -> Path:
    cid = lane["calibration"]["calibration_id"]
    shared = Path(env_path("CROPCOP_G2_SUMMARIES_DIR"))
    shared.mkdir(parents=True, exist_ok=True)
    dest = shared / f"{cid}.json"
    if dest.exists():
        return dest
    out = Path(env_path("CROPCOP_OUTPUT_ROOT")) / lane["lane_id"] / "calibration" / cid
    out.mkdir(parents=True, exist_ok=True)
    if lane["calibration"]["kind"] == "mnv4":
        idx = lane["seed_index"]
        evidence = Path(env_path("CROPCOP_G1_EVIDENCE_DIR"))
        pair_dir = Path(env_path("CROPCOP_PAIR_INIT_DIR"))
        cmd = [
            sys.executable, str(SCRIPTS / "calibrate.py"),
            "--config", lane["calibration"]["config"],
            "--condition", lane["calibration"]["condition"],
            "--pair-init", str(pair_dir / f"PAIR_INIT_S{idx}.pt"),
            "--pair-init-evidence", str(evidence / f"PAIR_INIT_S{idx}.json"),
            "--run-id", cid,
            "--output-dir", str(out),
            *common_args(source_sha, lane["lane_id"]),
        ]
        if lane["calibration"]["condition"] == "teacher":
            cmd += [
                "--teacher-checkpoint", env_path("CROPCOP_TEACHER_CHECKPOINT"),
                "--teacher-evidence", str(evidence / "DINO_TEACHER.json"),
                "--teacher-factory", env_path("CROPCOP_TEACHER_FACTORY"),
            ]
        cmd += durable_args(cid)
        run(cmd)
    else:
        run([
            sys.executable, str(SCRIPTS / "calibrate_cnxtt.py"),
            "--pretrained", env_path("CROPCOP_CNXTT_PRETRAINED"),
            "--output-dir", str(out),
            *common_args(source_sha, lane["lane_id"]),
        ])
    src = out / "calibration_summary.json"
    if not src.exists():
        raise RuntimeError(f"calibration did not create {src}")
    shutil.copy2(src, dest)
    publish_public_safe(cid, source_sha, [dest])
    return dest


def collect_g2_summaries_from_evidence_branches(shared_dir: Path) -> None:
    """Recover public-safe calibration summaries across independent Kaggle accounts."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    for cid in G2_IDS:
        dest = shared_dir / f"{cid}.json"
        if dest.exists():
            continue
        remote_ref = f"refs/remotes/origin/run-evidence/{cid}"
        fetch = subprocess.run(
            ["git", "fetch", "origin", f"refs/heads/run-evidence/{cid}:{remote_ref}"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if fetch.returncode != 0:
            continue
        repo_path = f"journal_extension/evidence/public/runs/{cid}/{cid}.json"
        show = subprocess.run(
            ["git", "show", f"{remote_ref}:{repo_path}"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if show.returncode == 0:
            tmp = dest.with_suffix(".json.tmp")
            tmp.write_text(show.stdout, encoding="utf-8")
            os.replace(tmp, dest)


def g2_barrier(shared_dir: Path, output_root: Path) -> Path | None:
    barrier = output_root / "G2_CALIBRATION_BARRIER.json"
    cp = run([
        sys.executable, str(SCRIPTS / "validate_g2_barrier.py"),
        "--summaries-dir", str(shared_dir),
        "--output", str(barrier),
    ], check=False)
    if cp.returncode != 0:
        print("G2 barrier not complete yet; this lane exits cleanly after persisting calibration evidence.")
        return None
    return barrier


def run_principal(lane: dict, source_sha: str, item: dict) -> dict:
    idx = lane["seed_index"]
    evidence = Path(env_path("CROPCOP_G1_EVIDENCE_DIR"))
    pair_dir = Path(env_path("CROPCOP_PAIR_INIT_DIR"))
    out = Path(env_path("CROPCOP_OUTPUT_ROOT")) / lane["lane_id"] / "principal" / item["run_id"]
    cmd = [
        sys.executable, str(SCRIPTS / "run_training.py"),
        "--config", item["config"],
        "--pair-init", str(pair_dir / f"PAIR_INIT_S{idx}.pt"),
        "--pair-init-evidence", str(evidence / f"PAIR_INIT_S{idx}.json"),
        "--run-id", item["run_id"],
        "--output-dir", str(out),
        "--resume-mode", "auto",
        *common_args(source_sha, lane["lane_id"]),
        *durable_args(item["run_id"]),
    ]
    if item["condition"] == "teacher":
        cmd += [
            "--teacher-checkpoint", env_path("CROPCOP_TEACHER_CHECKPOINT"),
            "--teacher-evidence", str(evidence / "DINO_TEACHER.json"),
            "--teacher-factory", env_path("CROPCOP_TEACHER_FACTORY"),
        ]
    run(cmd)
    record_path = out / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    evidence_files = [record_path]
    for candidate in (out / "metrics.json", out / "segments.jsonl"):
        if candidate.exists():
            evidence_files.append(candidate)
    publish_public_safe(item["run_id"], source_sha, evidence_files)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["K1", "K2", "K3"], default=os.environ.get("CROPCOP_LANE", ""))
    args = ap.parse_args()
    if not args.lane:
        raise SystemExit("select K1/K2/K3 with --lane or CROPCOP_LANE")
    missing = [x for x in REQUIRED_PATH_ENV if not os.environ.get(x)]
    missing += [x for x in COLUMN_ENV.values() if not os.environ.get(x)]
    if args.lane == "K3" and not os.environ.get("CROPCOP_CNXTT_PRETRAINED"):
        missing.append("CROPCOP_CNXTT_PRETRAINED")
    if missing:
        raise SystemExit("missing required environment variable names: " + ", ".join(sorted(set(missing))))

    lane = json.loads((KAGGLE / "lanes" / f"{args.lane}.json").read_text(encoding="utf-8"))
    source_sha = git_head()
    expected = os.environ.get("CROPCOP_SOURCE_GIT_COMMIT", "").strip()
    if expected and expected != source_sha:
        raise SystemExit(f"source commit drift: expected {expected}, observed {source_sha}")

    ensure_g1(lane)
    calibration(lane, source_sha)

    shared = Path(env_path("CROPCOP_G2_SUMMARIES_DIR"))
    collect_g2_summaries_from_evidence_branches(shared)
    output_root = Path(env_path("CROPCOP_OUTPUT_ROOT")) / args.lane
    barrier = g2_barrier(shared, output_root)
    if barrier is None:
        return 0
    barrier_data = json.loads(barrier.read_text(encoding="utf-8"))
    if barrier_data["source_git_commit"] != source_sha:
        raise SystemExit("G2 barrier source commit does not match this lane")

    for item in lane["principal"]:
        record = run_principal(lane, source_sha, item)
        if record["status"] == "LAUNCHED" and record.get("continuation_required"):
            print(f"{item['run_id']} reached a planned session boundary; rerun this identical lane to continue.")
            return 0
        if record["status"] != "PASS":
            raise SystemExit(f"principal run did not complete validly: {item['run_id']} status={record['status']}")
    print(f"{args.lane} principal pair is terminal PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
