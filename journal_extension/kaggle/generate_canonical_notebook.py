from __future__ import annotations

import json
from pathlib import Path

AUTHORIZED_SOURCE_SHA = "045fcf5c80366438b69a54288d9081e9b57ed973"

MARKDOWN = """# CropCop EAAI — Canonical Clean Kaggle Session

Thin orchestration only. For Stage 01A-SR use `smoke-write` in Saved Version A and `smoke-restore` in a completely fresh Saved Version B with the exact Smoke-A Notebook Output attached read-only. No CropCop dataset is required.
"""

CODE = r'''import os, platform, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path

# ============================================================
# CROPCOP EAAI — KAGGLE OPERATOR CONFIGURATION
# ============================================================
# Frozen Stage-01A-SR execution source. Do not change for this qualification.
AUTHORIZED_SOURCE_SHA = "045fcf5c80366438b69a54288d9081e9b57ed973"
LANE = os.environ.get("CROPCOP_LANE", "K1")
EXECUTION_PHASE = os.environ.get("CROPCOP_EXECUTION_PHASE", "smoke-write")
# Stage 01A-SR qualification phases: smoke-write | smoke-restore
# Future phases retained but not authorized by this stage: g1 | calibration | principal
REPOSITORY_URL = os.environ.get(
    "CROPCOP_REPOSITORY_URL",
    "https://github.com/rana-m-ahmed/ResearchWork-CropCop.git",
)
REPO_WORKDIR = os.environ.get("CROPCOP_REPO_WORKDIR", "/kaggle/working/cropcop-je")
OUTPUT_ROOT = os.environ.get("CROPCOP_OUTPUT_ROOT", "/kaggle/working/cropcop-je-output")
SYNTHETIC_SMOKE_ROOT = os.environ.get(
    "CROPCOP_SYNTHETIC_SMOKE_ROOT",
    "/kaggle/working/cropcop-smoke-input",
)
SMOKE_A_EXPORT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_A_EXPORT_ROOT",
    "/kaggle/working/cropcop-smoke-a-export",
)
SMOKE_B_EXPORT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_B_EXPORT_ROOT",
    "/kaggle/working/cropcop-smoke-b-export",
)
SMOKE_A_INPUT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_A_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_SMOKE_A_OUTPUT>",
)

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"] = repr(time.monotonic())
os.environ.setdefault("CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS", str(12 * 3600))
os.environ.setdefault("CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS", str(3600))
os.environ["CROPCOP_SOURCE_GIT_COMMIT"] = AUTHORIZED_SOURCE_SHA
os.environ["CROPCOP_LANE"] = LANE
os.environ["CROPCOP_EXECUTION_PHASE"] = EXECUTION_PHASE
os.environ["CROPCOP_OUTPUT_ROOT"] = OUTPUT_ROOT
os.environ["CROPCOP_SYNTHETIC_SMOKE_ROOT"] = SYNTHETIC_SMOKE_ROOT
os.environ["CROPCOP_SMOKE_A_EXPORT_ROOT"] = SMOKE_A_EXPORT_ROOT
os.environ["CROPCOP_SMOKE_B_EXPORT_ROOT"] = SMOKE_B_EXPORT_ROOT

assert platform.python_version() == "3.12.13", (
    f"Python re-lock required: {platform.python_version()}"
)
assert len(AUTHORIZED_SOURCE_SHA) == 40
assert all(c in "0123456789abcdef" for c in AUTHORIZED_SOURCE_SHA.lower())
assert LANE in {"K1", "K2", "K3"}
assert EXECUTION_PHASE in {"smoke-write", "smoke-restore", "g1", "calibration", "principal"}

repo_workdir = Path(REPO_WORKDIR).resolve()
for _mutable in (
    OUTPUT_ROOT,
    SYNTHETIC_SMOKE_ROOT,
    SMOKE_A_EXPORT_ROOT,
    SMOKE_B_EXPORT_ROOT,
):
    _m = Path(_mutable).resolve()
    assert _m != repo_workdir and repo_workdir not in _m.parents, (
        f"mutable output must be outside Git checkout: {_m}"
    )

from kaggle_secrets import UserSecretsClient

_secrets = UserSecretsClient()
_required_secrets = ["CROPCOP_GITHUB_TOKEN"]
if EXECUTION_PHASE in {"g1", "calibration", "principal"}:
    _required_secrets += ["KAGGLE_USERNAME", "KAGGLE_KEY"]

for _key in _required_secrets:
    if not os.environ.get(_key):
        try:
            os.environ[_key] = _secrets.get_secret(_key)
        except Exception as _exc:
            raise RuntimeError(f"Required Kaggle Secret missing: {_key}") from _exc

if repo_workdir.exists():
    shutil.rmtree(repo_workdir)

with tempfile.TemporaryDirectory() as td:
    askpass = Path(td) / "askpass.py"
    askpass.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "p=sys.argv[1] if len(sys.argv)>1 else ''\n"
        "print('x-access-token' if 'Username' in p else os.environ['CROPCOP_GITHUB_TOKEN'])\n"
    )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            REPOSITORY_URL,
            str(repo_workdir),
        ],
        env=env,
        check=True,
    )

subprocess.run(
    ["git", "-C", str(repo_workdir), "checkout", "--detach", AUTHORIZED_SOURCE_SHA],
    check=True,
)
actual = subprocess.check_output(
    ["git", "-C", str(repo_workdir), "rev-parse", "HEAD"],
    text=True,
).strip()
assert actual == AUTHORIZED_SOURCE_SHA, (
    f"HEAD mismatch: expected {AUTHORIZED_SOURCE_SHA}, got {actual}"
)
status = subprocess.check_output(
    ["git", "-C", str(repo_workdir), "status", "--porcelain=v1", "--untracked-files=all"],
    text=True,
)
assert not status.strip(), f"Git checkout is not clean: {status[:1000]}"

lockfile = repo_workdir / "journal_extension/requirements-training.lock.txt"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "-r",
        str(lockfile),
    ],
    check=True,
)

bootstrap_out = Path(OUTPUT_ROOT) / "bootstrap" / "clean_session.json"
bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        str(repo_workdir / "journal_extension/kaggle/bootstrap_clean_session.py"),
        "--repo-root",
        str(repo_workdir),
        "--authorized-source-sha",
        AUTHORIZED_SOURCE_SHA,
        "--phase",
        EXECUTION_PHASE,
        "--output",
        str(bootstrap_out),
    ],
    cwd=repo_workdir,
    check=True,
)

if EXECUTION_PHASE in {"smoke-write", "smoke-restore"}:
    cmd = [
        sys.executable,
        str(repo_workdir / "journal_extension/scripts/smoke_infrastructure.py"),
        "--mode",
        "write" if EXECUTION_PHASE == "smoke-write" else "restore",
        "--repo-root",
        str(repo_workdir),
        "--authorized-source-sha",
        AUTHORIZED_SOURCE_SHA,
        "--lane",
        LANE,
        "--runtime-output-root",
        OUTPUT_ROOT,
        "--synthetic-root",
        SYNTHETIC_SMOKE_ROOT,
        "--smoke-a-export-root",
        SMOKE_A_EXPORT_ROOT,
        "--smoke-b-export-root",
        SMOKE_B_EXPORT_ROOT,
    ]
    if EXECUTION_PHASE == "smoke-restore":
        if SMOKE_A_INPUT_ROOT.startswith("<"):
            raise RuntimeError(
                "Set SMOKE_A_INPUT_ROOT to the attached Smoke-A /kaggle/input/... path"
            )
        cmd += ["--smoke-a-input-root", SMOKE_A_INPUT_ROOT]
    subprocess.run(cmd, cwd=repo_workdir, check=True)
elif EXECUTION_PHASE == "g1":
    subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/kaggle/run_g1.py")],
        cwd=repo_workdir,
        check=True,
    )
else:
    subprocess.run(
        [
            sys.executable,
            str(repo_workdir / "journal_extension/kaggle/run_lane.py"),
            "--lane",
            LANE,
            "--phase",
            EXECUTION_PHASE,
        ],
        cwd=repo_workdir,
        check=True,
    )
'''


def _source_lines(source: str) -> list[str]:
    # Jupyter-compatible source serialization: physical Python lines with LF endings.
    return source.splitlines(keepends=True)


def build_notebook() -> dict:
    compile(CODE, "canonical_lane.ipynb", "exec")
    if len(CODE.splitlines()) <= 40:
        raise RuntimeError("canonical notebook source unexpectedly collapsed")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": _source_lines(MARKDOWN),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(CODE),
            },
        ],
    }


def main() -> int:
    target = Path(__file__).with_name("canonical_lane.ipynb")
    notebook = build_notebook()
    target.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
