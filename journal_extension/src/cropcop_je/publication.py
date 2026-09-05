from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


class PublicationError(RuntimeError):
    pass


MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt"}
DENIED_SUFFIXES = {".pt", ".pth", ".ckpt", ".pte", ".safetensors", ".bin", ".zip", ".tar", ".gz", ".7z", ".env"}
DENIED_NAME_PARTS = ("checkpoint", "raw_image", "prediction_logits", "token", "secret", "credential")
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"KAGGLE_KEY\s*[=:]\s*\S+", re.I),
    re.compile(r"(?:api[_-]?key|token|password|secret)\s*[=:]\s*['\"]?[^\s'\"]{8,}", re.I),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"/kaggle/input/[^\s\"']+"),
    re.compile(r"/kaggle/working/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\s\"']+", re.I),
)


def audit_public_files(paths: Iterable[Path]) -> list[Path]:
    approved = []
    for path in paths:
        if not path.is_file():
            raise PublicationError(f"public evidence path is not a file: {path}")
        if path.suffix.lower() in DENIED_SUFFIXES or path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise PublicationError(f"file type is not public-evidence allowlisted: {path}")
        low = path.name.lower()
        if any(part in low for part in DENIED_NAME_PARTS):
            raise PublicationError(f"file name looks restricted: {path.name}")
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            raise PublicationError(f"public evidence file exceeds {MAX_PUBLIC_FILE_BYTES} bytes: {path}")
        text = path.read_text(encoding="utf-8", errors="strict")
        for pattern in SECRET_PATTERNS + PRIVATE_PATH_PATTERNS:
            if pattern.search(text):
                raise PublicationError(f"public evidence audit rejected {path.name}: sensitive pattern {pattern.pattern!r}")
        approved.append(path)
    return approved


def _run_git(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=300)


def publish_to_github_branch(
    *,
    repo_dir: str | Path,
    source_git_sha: str,
    run_id: str,
    files: list[str | Path],
    destination_prefix: str = "journal_extension/evidence/public/runs",
    remote_name: str = "origin",
) -> str:
    token = os.environ.get("CROPCOP_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise PublicationError("Git credential absent; CROPCOP_GITHUB_TOKEN/GITHUB_TOKEN not set")
    approved = audit_public_files([Path(p) for p in files])
    repo_dir = Path(repo_dir).resolve()
    branch = f"run-evidence/{run_id}"
    with tempfile.TemporaryDirectory() as td:
        worktree = Path(td) / "worktree"
        remote_ref = f"refs/remotes/{remote_name}/{branch}"
        remote_exists = False
        try:
            _run_git(["git", "-C", str(repo_dir), "fetch", remote_name, f"refs/heads/{branch}:{remote_ref}"])
            remote_exists = True
        except subprocess.CalledProcessError:
            remote_exists = False
        base = f"{remote_name}/{branch}" if remote_exists else source_git_sha
        _run_git(["git", "-C", str(repo_dir), "worktree", "add", "--detach", str(worktree), base])
        try:
            _run_git(["git", "-C", str(worktree), "checkout", "-B", branch])
            if remote_exists:
                ancestor = subprocess.run(
                    ["git", "-C", str(worktree), "merge-base", "--is-ancestor", source_git_sha, "HEAD"],
                    capture_output=True,
                    text=True,
                )
                if ancestor.returncode != 0:
                    raise PublicationError("existing evidence branch is not descended from the authorized source commit")
            dest_root = worktree / destination_prefix / run_id
            dest_root.mkdir(parents=True, exist_ok=True)
            for src in approved:
                shutil.copy2(src, dest_root / src.name)
            staged = [str((Path(destination_prefix) / run_id / p.name).as_posix()) for p in approved]
            _run_git(["git", "-C", str(worktree), "add", "--", *staged])
            diff = _run_git(["git", "-C", str(worktree), "diff", "--cached", "--name-only"]).stdout.splitlines()
            if sorted(diff) != sorted(staged):
                raise PublicationError(f"staged file set differs from explicit allowlist: staged={diff}, expected={staged}")
            if not diff:
                return branch
            _run_git(["git", "-C", str(worktree), "commit", "-m", f"evidence({run_id}): publish public-safe run evidence"])
            askpass = Path(td) / "askpass.sh"
            askpass.write_text('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token ;; *) printf "%s\\n" "$CROPCOP_GITHUB_TOKEN" ;; esac\n')
            askpass.chmod(0o700)
            env = dict(os.environ)
            env["CROPCOP_GITHUB_TOKEN"] = token
            env["GIT_ASKPASS"] = str(askpass)
            env["GIT_TERMINAL_PROMPT"] = "0"
            _run_git(["git", "-C", str(worktree), "push", remote_name, f"HEAD:refs/heads/{branch}"], env=env)
            return branch
        finally:
            subprocess.run(["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(worktree)], check=False, capture_output=True, text=True)
