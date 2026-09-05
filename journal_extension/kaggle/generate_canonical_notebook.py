from __future__ import annotations

import json
from pathlib import Path

MGPU_EXECUTION_SOURCE_SHA = "be9b6965d760ff6e8674623b658f66572cd57093"
AUTHORIZED_SOURCE_SHA = "be9b6965d760ff6e8674623b658f66572cd57093"

MARKDOWN = """# CropCop EAAI — Canonical Stage-01A-MGPU Kaggle Wrapper

Thin orchestration only, hard-bound to the frozen Stage-01A-MGPU-QA1 execution source.

Operator phases: `smoke-write`, `smoke-restore`, `dual-gpu-smoke`, `g1`, `calibration-dual`, `principal-dual`.

For `principal-dual`, set non-secret `CROPCOP_PRINCIPAL_ENVELOPE` to `P1`, `P2`, or `P3`. The notebook delegates the fixed experiment mapping to repository envelope configs.

Smoke A/B remain cross-Saved-Version and API-free. `smoke-restore` requires the exact Smoke-A Notebook Output attached read-only.
"""

CODE = r'''import json
import os, platform, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# ============================================================
# CROPCOP EAAI — KAGGLE OPERATOR CONFIGURATION
# ============================================================
# Frozen Stage-01A-MGPU execution source. Wrapper commits are not execution sources.
AUTHORIZED_SOURCE_SHA = "be9b6965d760ff6e8674623b658f66572cd57093"
LANE = os.environ.get("CROPCOP_LANE", "K1")
EXECUTION_PHASE = os.environ.get("CROPCOP_EXECUTION_PHASE", "smoke-write")
PRINCIPAL_ENVELOPE = os.environ.get("CROPCOP_PRINCIPAL_ENVELOPE", "P1").strip().upper()
# Exact operator phases: smoke-write | smoke-restore | dual-gpu-smoke | g1 | calibration-dual | principal-dual
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
SMOKE_B_INPUT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_B_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_SMOKE_B_OUTPUT>",
)
DUAL_GPU_SMOKE_INPUT_ROOT = os.environ.get(
    "CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_DUAL_GPU_SMOKE_OUTPUT>",
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
assert EXECUTION_PHASE in {
    "smoke-write",
    "smoke-restore",
    "dual-gpu-smoke",
    "g1",
    "calibration-dual",
    "principal-dual",
}
if EXECUTION_PHASE == "principal-dual":
    if PRINCIPAL_ENVELOPE not in {"P1", "P2", "P3"}:
        raise RuntimeError(
            "CROPCOP_PRINCIPAL_ENVELOPE must be exactly P1, P2, or P3 for principal-dual"
        )
    os.environ["CROPCOP_ENVELOPE_ID"] = PRINCIPAL_ENVELOPE

def _locate_exact_attached_evidence(root_value: str, filename: str, label: str) -> str:
    if not root_value or root_value.startswith("<"):
        raise RuntimeError(f"Set {label} to the exact attached Notebook Output root")
    root = Path(root_value).resolve()
    if not root.is_dir():
        raise RuntimeError(f"{label} does not exist or is not a directory: {root}")
    matches = sorted(path.resolve() for path in root.rglob(filename) if path.is_file())
    matches = [path for path in matches if path == root or root in path.parents]
    if len(matches) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one {filename}; found {len(matches)} below explicit root {root}"
        )
    return str(matches[0])


if EXECUTION_PHASE == "dual-gpu-smoke":
    os.environ["CROPCOP_INFRA_SMOKE_EVIDENCE"] = _locate_exact_attached_evidence(
        SMOKE_B_INPUT_ROOT,
        "SMOKE_B_EVIDENCE.json",
        "CROPCOP_SMOKE_B_INPUT_ROOT",
    )
elif EXECUTION_PHASE in {"g1", "calibration-dual", "principal-dual"}:
    os.environ["CROPCOP_INFRA_SMOKE_EVIDENCE"] = _locate_exact_attached_evidence(
        SMOKE_B_INPUT_ROOT,
        "SMOKE_B_EVIDENCE.json",
        "CROPCOP_SMOKE_B_INPUT_ROOT",
    )
    os.environ["CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"] = _locate_exact_attached_evidence(
        DUAL_GPU_SMOKE_INPUT_ROOT,
        "DUAL_GPU_SMOKE_EVIDENCE.json",
        "CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT",
    )


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
if EXECUTION_PHASE in {"g1", "calibration-dual", "principal-dual"}:
    _required_secrets += ["KAGGLE_USERNAME", "KAGGLE_KEY"]

for _key in _required_secrets:
    if not os.environ.get(_key):
        try:
            os.environ[_key] = _secrets.get_secret(_key)
        except Exception as _exc:
            raise RuntimeError(f"Required Kaggle Secret missing: {_key}") from _exc

_token = str(os.environ.get("CROPCOP_GITHUB_TOKEN", "")).strip()
if not _token:
    raise RuntimeError("CROPCOP_GITHUB_TOKEN is empty after Kaggle Secrets retrieval")
if any(ch.isspace() for ch in _token):
    raise RuntimeError(
        "CROPCOP_GITHUB_TOKEN contains whitespace/newlines; store the raw token only"
    )
if (_token.startswith("'") and _token.endswith("'")) or (
    _token.startswith('"') and _token.endswith('"')
):
    raise RuntimeError(
        "CROPCOP_GITHUB_TOKEN appears to include surrounding quotes; store the raw token only"
    )
os.environ["CROPCOP_GITHUB_TOKEN"] = _token

_repo_parts = urlparse(REPOSITORY_URL)
if _repo_parts.scheme != "https" or _repo_parts.netloc.lower() != "github.com":
    raise RuntimeError("Stage 01A-MGPU requires an HTTPS github.com repository URL")
_repo_segments = [part for part in _repo_parts.path.strip("/").split("/") if part]
if len(_repo_segments) != 2:
    raise RuntimeError(f"Unexpected GitHub repository URL path: {_repo_parts.path}")
_repo_owner = _repo_segments[0]
_repo_name = _repo_segments[1][:-4] if _repo_segments[1].endswith(".git") else _repo_segments[1]
_git_username = _repo_owner
os.environ["CROPCOP_GIT_USERNAME"] = _git_username

_api_url = f"https://api.github.com/repos/{_repo_owner}/{_repo_name}"
_api_request = Request(
    _api_url,
    headers={
        "Authorization": f"Bearer {_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cropcop-kaggle-auth-preflight",
    },
)
try:
    with urlopen(_api_request, timeout=30) as _response:
        _repo_payload = json.loads(_response.read().decode("utf-8"))
except HTTPError as _exc:
    if _exc.code == 401:
        raise RuntimeError(
            "GitHub rejected CROPCOP_GITHUB_TOKEN (HTTP 401). "
            "The token is invalid, expired, revoked, or was copied incorrectly."
        ) from _exc
    if _exc.code == 404:
        raise RuntimeError(
            "GitHub could not expose the private repository to this token (HTTP 404). "
            "For a fine-grained PAT, select resource owner 'rana-m-ahmed', include "
            "repository 'ResearchWork-CropCop', and grant Contents: Read and write."
        ) from _exc
    if _exc.code == 403:
        raise RuntimeError(
            "GitHub denied the token by policy/permission (HTTP 403). "
            "Check token repository access, organization/SSO policy if applicable, and expiry."
        ) from _exc
    raise RuntimeError(f"GitHub repository authorization preflight failed with HTTP {_exc.code}") from _exc
except URLError as _exc:
    raise RuntimeError(f"GitHub authorization preflight network failure: {_exc.reason}") from _exc

if _repo_payload.get("full_name") != f"{_repo_owner}/{_repo_name}":
    raise RuntimeError("GitHub authorization preflight returned an unexpected repository identity")
if _repo_payload.get("private") is not True:
    raise RuntimeError("Expected CropCop repository to be private during smoke qualification")

print(
    "GitHub API auth preflight: PASS "
    f"(repo={_repo_owner}/{_repo_name}, token_value_not_printed=true)"
)

if repo_workdir.exists():
    shutil.rmtree(repo_workdir)

with tempfile.TemporaryDirectory() as td:
    askpass = Path(td) / "askpass.py"
    askpass.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "p=(sys.argv[1] if len(sys.argv)>1 else '').lower()\n"
        "if 'username' in p:\n"
        "    print(os.environ['CROPCOP_GIT_USERNAME'])\n"
        "elif 'password' in p:\n"
        "    print(os.environ['CROPCOP_GITHUB_TOKEN'])\n"
        "else:\n"
        "    raise SystemExit(2)\n"
    )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_ASKPASS_REQUIRE"] = "force"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "credential.helper"
    env["GIT_CONFIG_VALUE_0"] = ""

    _ls_remote = subprocess.run(
        ["git", "ls-remote", "--exit-code", REPOSITORY_URL, "HEAD"],
        env=env,
        capture_output=True,
        text=True,
    )
    if _ls_remote.returncode != 0:
        _safe_stderr = (_ls_remote.stderr or "").replace(_token, "<redacted>")
        raise RuntimeError(
            "GitHub API token validation passed, but Git-over-HTTPS read authentication failed. "
            "The token may lack repository Contents read permission. "
            f"git ls-remote stderr: {_safe_stderr[-1200:]}"
        )
    print("GitHub Git-over-HTTPS read preflight: PASS")

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
        env=env,
        check=True,
    )

    _probe_ref = f"refs/heads/run-evidence/auth-probe-{AUTHORIZED_SOURCE_SHA[:12]}"
    _push_probe = subprocess.run(
        [
            "git",
            "-C",
            str(repo_workdir),
            "push",
            "--dry-run",
            "origin",
            f"HEAD:{_probe_ref}",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if _push_probe.returncode != 0:
        _safe_stderr = (_push_probe.stderr or "").replace(_token, "<redacted>")
        raise RuntimeError(
            "Private clone/read succeeded, but GitHub evidence-branch write preflight failed. "
            "For a fine-grained PAT, grant Contents: Read and write on ResearchWork-CropCop. "
            f"git push --dry-run stderr: {_safe_stderr[-1200:]}"
        )
    print("GitHub evidence-branch write preflight: PASS (dry-run only; no ref created)")

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
elif EXECUTION_PHASE == "dual-gpu-smoke":
    subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/scripts/smoke_dual_gpu.py")],
        cwd=repo_workdir,
        check=True,
    )
elif EXECUTION_PHASE == "g1":
    subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/kaggle/run_g1.py")],
        cwd=repo_workdir,
        check=True,
    )
elif EXECUTION_PHASE in {"calibration-dual", "principal-dual"}:
    subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/kaggle/run_envelope.py")],
        cwd=repo_workdir,
        check=True,
    )
else:
    raise RuntimeError(f"unsupported canonical execution phase: {EXECUTION_PHASE}")
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
