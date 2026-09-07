from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .g1 import validate_dependency_lock_object
from .hashing import sha256_file, sha256_json
from .envelope import AMENDMENT_ID, AMENDMENT_SHA256, validate_envelope_config
from .science_diff import validate_science_diff
from .runlog import IDENTITY_FIELDS, validate_run_record
from .surfaces import validate_training_config

EXPECTED_AUTHORITY_SHA = "aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74"
EXPECTED_STAGE04_SHA = "a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079"
EXPECTED_AMENDMENT_ID = AMENDMENT_ID
EXPECTED_AMENDMENT_SHA = AMENDMENT_SHA256
EXPECTED_SEEDS = [21270083, 606135704, 1153870846]
EXPECTED_IDS = {
    *(f"R04-MNV4-DIRECT-S{i}" for i in (1, 2, 3)),
    *(f"R05-MNV4-TEACHER-S{i}" for i in (1, 2, 3)),
}
LOCKED_REQUIREMENTS = {
    "torch==2.12.1",
    "torchvision==0.27.1",
    "timm==1.0.26",
    "numpy==2.5.2",
    "Pillow==12.3.0",
    "safetensors==0.8.0",
    "kaggle==2.2.4",
    "huggingface-hub==1.30.0",
    "transformers==5.0.0",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _requirements(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def validate_static(repo_root: Path) -> dict:
    errors = []
    je = repo_root / "journal_extension"
    authority = _load(je / "locks/scientific_authority.json")
    registry = _load(je / "locks/experiment_registry.json")

    if authority.get("authority_sha256") != EXPECTED_AUTHORITY_SHA:
        errors.append("03R authority SHA mismatch")
    if authority.get("stage04_sha256") != EXPECTED_STAGE04_SHA:
        errors.append("Stage-04 architecture SHA mismatch")
    if authority.get("stage04_execution_amendment_id") != EXPECTED_AMENDMENT_ID:
        errors.append("Stage-04A execution amendment ID mismatch")
    if authority.get("stage04_execution_amendment_sha256") != EXPECTED_AMENDMENT_SHA:
        errors.append("Stage-04A execution amendment SHA mismatch")

    science = validate_science_diff(repo_root)
    errors.extend("science-diff: " + x for x in science.get("errors", []))

    principal = [x for x in registry["experiments"] if x["experiment_id"] in EXPECTED_IDS]
    if {x["experiment_id"] for x in principal} != EXPECTED_IDS:
        errors.append("principal R04/R05 ID set mismatch")
    by_id = {x["experiment_id"]: x for x in principal}
    for i, seed in enumerate(EXPECTED_SEEDS, 1):
        d = by_id.get(f"R04-MNV4-DIRECT-S{i}", {})
        t = by_id.get(f"R05-MNV4-TEACHER-S{i}", {})
        if d.get("seed") != seed or t.get("seed") != seed:
            errors.append(f"seed mismatch for S{i}")
        if d.get("pair_id") != t.get("pair_id"):
            errors.append(f"paired identity mismatch for S{i}")

    hashes = {}
    for condition, folder in (("direct", "r04_direct"), ("teacher", "r05_teacher")):
        for i in (1, 2, 3):
            path = je / "configs" / folder / f"s{i}.json"
            cfg = _load(path)
            try:
                validate_training_config(cfg)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
            expected = f"{'R04-MNV4-DIRECT' if condition == 'direct' else 'R05-MNV4-TEACHER'}-S{i}"
            if cfg.get("experiment_id") != expected:
                errors.append(f"config experiment ID mismatch: {path}")
            if cfg.get("seed") != EXPECTED_SEEDS[i - 1]:
                errors.append(f"config seed mismatch: {path}")
            hashes[expected] = sha256_json(cfg)
    for i in (1, 2, 3):
        d = _load(je / "configs/r04_direct" / f"s{i}.json")
        t = _load(je / "configs/r05_teacher" / f"s{i}.json")
        if d["required_student_init_evidence"] != t["required_student_init_evidence"]:
            errors.append(f"S{i} pair-init evidence mismatch")
        if d["pair_id"] != t["pair_id"]:
            errors.append(f"S{i} pair_id mismatch")

    lineage_root = je / "evidence/historical/teacher_stage1"
    lineage_path = lineage_root / "teacher_lineage_manifest.json"
    try:
        lineage = _load(lineage_path)
        sources = lineage.get("historical_evidence_sources", [])
        if not isinstance(sources, list) or not sources:
            errors.append("historical teacher lineage has no evidence sources")
        else:
            resolved_root = lineage_root.resolve()
            for row in sources:
                rel = str(row.get("path", ""))
                candidate = (resolved_root / rel).resolve()
                if resolved_root not in candidate.parents:
                    errors.append(f"historical teacher evidence path escapes root: {rel}")
                    continue
                if not candidate.is_file():
                    errors.append(f"historical teacher evidence source missing: {rel}")
                    continue
                expected = str(row.get("sha256", ""))
                actual = sha256_file(candidate)
                if not expected or actual != expected:
                    errors.append(
                        f"historical teacher evidence SHA mismatch: {rel}; "
                        f"expected={expected or '<missing>'}; actual={actual}"
                    )
    except Exception as exc:
        errors.append(f"historical teacher lineage validation failed: {exc}")

    dep_lock = _load(je / "locks/execution_dependency_lock.json")
    errors.extend(f"dependency lock: {x}" for x in validate_dependency_lock_object(dep_lock))
    for req_path in (je / "requirements-training.txt", je / "requirements-training.lock.txt"):
        if _requirements(req_path) != LOCKED_REQUIREMENTS:
            errors.append(f"execution requirements are not exactly frozen: {req_path.relative_to(repo_root)}")

    required_identity = {
        "g1_seal_sha256", "g2_barrier_sha256", "dependency_lock_sha256",
        "teacher_factory_bundle_sha256", "source_git_commit", "lane_id",
    }
    if not required_identity.issubset(set(IDENTITY_FIELDS)):
        errors.append("run/resume identity does not include all Stage-01A-P G1/G2 fields")

    notebook_path = je / "kaggle/canonical_lane.ipynb"
    try:
        notebook = _load(notebook_path)
        code_cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
        if notebook.get("nbformat") != 4 or len(code_cells) != 1:
            errors.append("canonical Kaggle notebook must be one thin nbformat-4 code cell")
        else:
            code = "".join(code_cells[0].get("source", []))
            for token in (
                "CROPCOP_NOTEBOOK_STARTED_MONOTONIC",
                "PYTHONDONTWRITEBYTECODE",
                "GIT_ASKPASS",
                "GIT_ASKPASS_REQUIRE",
                "GitHub API auth preflight: PASS",
                "GitHub Git-over-HTTPS read preflight: PASS",
                "GitHub evidence-branch write preflight: PASS",
                '"ls-remote"',
                '"--dry-run"',
                "run-evidence/auth-probe-",
                '"checkout"',
                '"--detach"',
                "run_g1.py",
                "run_envelope.py",
                "smoke_infrastructure.py",
                "smoke_dual_gpu.py",
                "smoke-write",
                "smoke-restore",
                "dual-gpu-smoke",
                "g1",
                "calibration-dual",
                "principal-dual",
                "CROPCOP_PRINCIPAL_ENVELOPE",
                '"P1"',
                '"P2"',
                '"P3"',
                "SMOKE_A_INPUT_ROOT",
                "CROPCOP_RFDV_ROOT",
                "CROPCOP_FINAL_V1_ROOT",
                "CROPCOP_G1_PRIVATE_DATASET_SLUG",
                "CROPCOP_G1_INPUT_ROOT",
            ):
                if token not in code:
                    errors.append(f"canonical Kaggle bootstrap missing: {token}")
            clock_index = code.find("CROPCOP_NOTEBOOK_STARTED_MONOTONIC")
            clone_index = code.find('"clone"')
            pip_index = code.find('"pip"')
            if clone_index < 0:
                errors.append("canonical Kaggle notebook clone command token is missing")
            if pip_index < 0:
                errors.append("canonical Kaggle notebook pip-install command token is missing")
            if clock_index < 0 or (clone_index >= 0 and clock_index > clone_index) or (pip_index >= 0 and clock_index > pip_index):
                errors.append("notebook-global clock is not established before clone/install work")
            try:
                compile(code, "canonical_lane.ipynb", "exec")
            except SyntaxError as exc:
                errors.append(f"canonical Kaggle notebook code does not compile: {exc}")
            lines = code.splitlines()
            if len(lines) <= 100:
                errors.append(
                    f"canonical Kaggle notebook source is not physically multiline enough: {len(lines)} lines"
                )
            expected_prefix = [
                "import json",
                "import os, platform, shutil, stat, subprocess, sys, tempfile, time",
                "from pathlib import Path",
                "from urllib.error import HTTPError, URLError",
                "from urllib.parse import urlparse",
                "from urllib.request import Request, urlopen",
            ]
            if lines[:len(expected_prefix)] != expected_prefix:
                errors.append("canonical Kaggle notebook Python import prefix is unexpected")
            expected_source_binding = (
                'AUTHORIZED_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"'
            )
            if expected_source_binding not in lines:
                errors.append("canonical Kaggle notebook frozen execution-source binding mismatch")
            for forbidden in (
                "EXECUTION_PHASE == 'smoke'",
                'EXECUTION_PHASE == "smoke"',
                "phase == 'smoke'",
                'phase == "smoke"',
                'EXECUTION_PHASE == "calibration"',
                'EXECUTION_PHASE == "principal"',
                '"--phase", "calibration"',
                '"--phase", "principal"',
            ):
                if forbidden in code:
                    errors.append(f"ambiguous/legacy canonical phase route remains: {forbidden}")
    except Exception as exc:
        errors.append(f"canonical Kaggle notebook invalid: {exc}")

    runner = (je / "kaggle/run_lane.py").read_text(encoding="utf-8")
    if "prepare_pair_init.py" in runner or "save_pair_initialization" in runner:
        errors.append("production lane runner can regenerate pair initialization after G1")
    for token in ("validate_g1_barrier.py", "validate_prelaunch.py", "--g1-seal", "--g2-barrier"):
        if token not in runner:
            errors.append(f"production lane runner missing hard gate token: {token}")
    for forbidden in ("DistributedDataParallel", "nn.DataParallel", "SyncBatchNorm"):
        if forbidden in runner:
            errors.append(f"production lane runner contains forbidden MGPU scientific path: {forbidden}")
    for token in ("CROPCOP_DUAL_ENVELOPE", "torch.cuda.device_count() != 1", "CROPCOP_NUM_WORKERS_PER_CHILD"):
        if token not in runner:
            errors.append(f"production lane runner missing MGPU child-isolation token: {token}")

    for envelope_name in ("G2_DUAL_T4.json", "P1_S1_PAIR.json", "P2_S2_PAIR.json", "P3_S3_PAIR.json"):
        envelope_path = je / "kaggle/envelopes" / envelope_name
        if not envelope_path.is_file():
            errors.append(f"MGPU envelope config missing: {envelope_name}")
            continue
        errors.extend(
            f"{envelope_name}: {message}"
            for message in validate_envelope_config(_load(envelope_path))
        )

    bootstrap = (je / "kaggle/bootstrap_clean_session.py").read_text(encoding="utf-8")
    for phase in ("dual-gpu-smoke", "calibration-dual", "principal-dual"):
        if phase not in bootstrap:
            errors.append(f"clean-session bootstrap missing MGPU phase: {phase}")

    for i in (1, 2, 3):
        lane = _load(je / "kaggle/lanes" / f"K{i}.json")
        for item in lane.get("principal", []):
            if "run_id" in item:
                errors.append(f"K{i} lane predeclares a principal run ID before launch")

    for required in (
        "scripts/capture_mnv4_pretrained_provenance.py",
        "scripts/capture_teacher_factory_bundle.py",
        "scripts/verify_teacher_class_order.py",
        "scripts/seal_g1.py",
        "scripts/validate_g1_barrier.py",
        "scripts/smoke_infrastructure.py",
        "scripts/smoke_dual_gpu.py",
        "scripts/check_science_diff.py",
        "src/cropcop_je/envelope.py",
        "src/cropcop_je/science_diff.py",
        "kaggle/bootstrap_clean_session.py",
        "kaggle/run_g1.py",
        "kaggle/run_envelope.py",
    ):
        if not (je / required).is_file():
            errors.append(f"Stage-01A-P execution component missing: journal_extension/{required}")

    try:
        eol_output = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-files", "--eol"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        eol_offenders = []
        for line in eol_output.splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            fields = meta.split()
            index_eol = fields[0] if fields else ""
            if "eol=lf" in meta and index_eol in {"i/crlf", "i/mixed"}:
                eol_offenders.append(f"{path} ({index_eol}; {meta})")
        if eol_offenders:
            errors.append(
                "tracked LF text blobs are not canonically normalized: "
                + "; ".join(eol_offenders)
            )
    except Exception as exc:
        errors.append(f"unable to validate tracked text EOL canonicalization: {exc}")

    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for token in ("*.pt", "*.pth", "*.ckpt", "*.pte", "*.safetensors", ".env"):
        if token not in gitignore:
            errors.append(f".gitignore lacks restricted-artifact guard {token}")

    runs = je / "runs"
    if runs.exists():
        for p in runs.glob("*.json"):
            try:
                validate_run_record(_load(p))
            except Exception as exc:
                errors.append(f"invalid run record {p.name}: {exc}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "authority_sha256": authority.get("authority_sha256"),
        "dependency_lock_sha256": dep_lock.get("dependency_lock_sha256"),
        "stage04_execution_amendment_id": authority.get("stage04_execution_amendment_id"),
        "stage04_execution_amendment_sha256": authority.get("stage04_execution_amendment_sha256"),
        "science_diff_status": science.get("status"),
        "principal_config_sha256": hashes,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--json-report", default="")
    args = ap.parse_args()
    report = validate_static(Path(args.repo_root).resolve())
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_report:
        Path(args.json_report).write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
