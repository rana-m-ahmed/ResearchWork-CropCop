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

EVIDENCE_AUTHOR_NAME = "CropCop Evidence Publisher"
EVIDENCE_AUTHOR_EMAIL = "cropcop-evidence@localhost.invalid"


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
                raise PublicationError(
                    f"public evidence audit rejected {path.name}: sensitive pattern {pattern.pattern!r}"
                )
        approved.append(path)
    return approved


def _redact(text: str, env: dict[str, str] | None) -> str:
    safe = text or ""
    candidates = []
    if env:
        candidates.extend(
            [
                env.get("CROPCOP_GITHUB_TOKEN", ""),
                env.get("GITHUB_TOKEN", ""),
            ]
        )
    candidates.extend(
        [
            os.environ.get("CROPCOP_GITHUB_TOKEN", ""),
            os.environ.get("GITHUB_TOKEN", ""),
        ]
    )
    for secret in candidates:
        if secret:
            safe = safe.replace(secret, "<redacted>")
    return safe


def _raise_git_failure(
    args: list[str],
    result: subprocess.CompletedProcess[str],
    *,
    env: dict[str, str] | None,
    context: str,
) -> None:
    stderr = _redact((result.stderr or "").strip(), env)
    stdout = _redact((result.stdout or "").strip(), env)
    detail = stderr or stdout or "<no git diagnostic text>"
    if len(detail) > 2000:
        detail = detail[-2000:]
    command = " ".join(args)
    raise PublicationError(
        f"{context} failed (git rc={result.returncode}; command={command}): {detail}"
    )


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    context: str = "git command",
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise PublicationError(f"{context} timed out after 300 seconds") from exc
    if check and result.returncode != 0:
        _raise_git_failure(args, result, env=env, context=context)
    return result


def _publication_env(token: str) -> dict[str, str]:
    token = token.strip()
    if not token:
        raise PublicationError("Git credential is empty")
    if any(ch.isspace() for ch in token):
        raise PublicationError("Git credential contains whitespace/newlines")
    env = dict(os.environ)
    env["CROPCOP_GITHUB_TOKEN"] = token
    env["CROPCOP_GIT_USERNAME"] = (
        os.environ.get("CROPCOP_GIT_USERNAME", "").strip() or "x-access-token"
    )
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS_REQUIRE"] = "force"
    # Evidence commits must work in clean ephemeral hosts with no global Git identity.
    env["GIT_AUTHOR_NAME"] = EVIDENCE_AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = EVIDENCE_AUTHOR_EMAIL
    env["GIT_COMMITTER_NAME"] = EVIDENCE_AUTHOR_NAME
    env["GIT_COMMITTER_EMAIL"] = EVIDENCE_AUTHOR_EMAIL
    # Do not inherit an arbitrary credential helper from the Kaggle base image.
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "credential.helper"
    env["GIT_CONFIG_VALUE_0"] = ""
    return env


def _verify_destination_snapshot(dest_root: Path, approved: list[Path]) -> None:
    expected_names = [path.name for path in approved]
    if len(expected_names) != len(set(expected_names)):
        raise PublicationError(
            f"duplicate public evidence basenames would overwrite each other: {expected_names}"
        )

    observed_names = sorted(
        path.relative_to(dest_root).as_posix()
        for path in dest_root.rglob("*")
        if path.is_file()
    )
    expected_sorted = sorted(expected_names)
    if observed_names != expected_sorted:
        raise PublicationError(
            "destination public-evidence snapshot differs from explicit allowlist: "
            f"observed={observed_names}, expected={expected_sorted}"
        )

    for src in approved:
        dst = dest_root / src.name
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            raise PublicationError(
                f"destination public-evidence bytes differ from approved source: {src.name}"
            )


def _verify_staged_subset(changed_paths: list[str], allowed_paths: list[str]) -> None:
    unexpected = sorted(set(changed_paths) - set(allowed_paths))
    if unexpected:
        raise PublicationError(
            "staged file set contains paths outside explicit allowlist: "
            f"unexpected={unexpected}, allowed={sorted(allowed_paths)}"
        )


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
    approved_names = [path.name for path in approved]
    if len(approved_names) != len(set(approved_names)):
        raise PublicationError(
            f"duplicate public evidence basenames would overwrite each other: {approved_names}"
        )

    repo_dir = Path(repo_dir).resolve()
    branch = f"run-evidence/{run_id}"

    with tempfile.TemporaryDirectory() as td:
        worktree = Path(td) / "worktree"
        askpass = Path(td) / "askpass.py"
        askpass.write_text(
            "#!/usr/bin/env python3\n"
            "import os,sys\n"
            "p=(sys.argv[1] if len(sys.argv)>1 else '').lower()\n"
            "if 'username' in p:\n"
            "    print(os.environ.get('CROPCOP_GIT_USERNAME','x-access-token'))\n"
            "elif 'password' in p:\n"
            "    print(os.environ['CROPCOP_GITHUB_TOKEN'])\n"
            "else:\n"
            "    raise SystemExit(2)\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)

        env = _publication_env(token)
        env["GIT_ASKPASS"] = str(askpass)

        remote_ref = f"refs/remotes/{remote_name}/{branch}"
        remote_head = f"refs/heads/{branch}"

        probe = _run_git(
            ["git", "-C", str(repo_dir), "ls-remote", "--exit-code", "--heads", remote_name, remote_head],
            env=env,
            check=False,
            context="evidence branch existence probe",
        )
        if probe.returncode == 0:
            remote_exists = True
        elif probe.returncode == 2:
            remote_exists = False
        else:
            _raise_git_failure(
                ["git", "-C", str(repo_dir), "ls-remote", "--exit-code", "--heads", remote_name, remote_head],
                probe,
                env=env,
                context="evidence branch existence probe",
            )

        if remote_exists:
            _run_git(
                ["git", "-C", str(repo_dir), "fetch", remote_name, f"{remote_head}:{remote_ref}"],
                env=env,
                context="evidence branch fetch",
            )

        base = f"{remote_name}/{branch}" if remote_exists else source_git_sha
        _run_git(
            ["git", "-C", str(repo_dir), "worktree", "add", "--detach", str(worktree), base],
            env=env,
            context="evidence worktree creation",
        )
        try:
            _run_git(
                ["git", "-C", str(worktree), "checkout", "-B", branch],
                env=env,
                context="evidence branch checkout",
            )
            if remote_exists:
                ancestor = _run_git(
                    ["git", "-C", str(worktree), "merge-base", "--is-ancestor", source_git_sha, "HEAD"],
                    env=env,
                    check=False,
                    context="evidence branch ancestry check",
                )
                if ancestor.returncode == 1:
                    raise PublicationError(
                        "existing evidence branch is not descended from the authorized source commit"
                    )
                if ancestor.returncode != 0:
                    _raise_git_failure(
                        ["git", "-C", str(worktree), "merge-base", "--is-ancestor", source_git_sha, "HEAD"],
                        ancestor,
                        env=env,
                        context="evidence branch ancestry check",
                    )

            dest_root = worktree / destination_prefix / run_id
            dest_root.mkdir(parents=True, exist_ok=True)
            for src in approved:
                shutil.copy2(src, dest_root / src.name)

            # The worktree snapshot must contain exactly the approved public files and
            # byte-match their audited sources. This is the security invariant.
            _verify_destination_snapshot(dest_root, approved)

            staged = [
                str((Path(destination_prefix) / run_id / p.name).as_posix())
                for p in approved
            ]
            _run_git(
                ["git", "-C", str(worktree), "add", "--", *staged],
                env=env,
                context="evidence staging",
            )
            diff = _run_git(
                ["git", "-C", str(worktree), "diff", "--cached", "--name-only"],
                env=env,
                context="evidence staged-file verification",
            ).stdout.splitlines()

            # A pre-existing evidence branch can already contain one or more approved
            # files with identical bytes. Git correctly omits those unchanged paths
            # from the staged diff, so the diff is allowed to be a subset of the
            # explicit allowlist. Any changed path outside the allowlist still fails
            # closed.
            _verify_staged_subset(diff, staged)
            if not diff:
                return branch

            _run_git(
                [
                    "git",
                    "-C",
                    str(worktree),
                    "commit",
                    "-m",
                    f"evidence({run_id}): publish public-safe run evidence",
                ],
                env=env,
                context="public evidence commit",
            )
            _run_git(
                ["git", "-C", str(worktree), "push", remote_name, f"HEAD:{remote_head}"],
                env=env,
                context="public evidence push",
            )
            return branch
        finally:
            subprocess.run(
                ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(worktree)],
                check=False,
                capture_output=True,
                text=True,
            )
