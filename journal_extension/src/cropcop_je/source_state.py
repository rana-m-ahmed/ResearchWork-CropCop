from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class SourceStateError(RuntimeError):
    pass


def _git(repo_root: Path, *args: str) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return cp.stdout.strip()


def is_within(path: str | Path, root: str | Path) -> bool:
    p = Path(path).resolve()
    r = Path(root).resolve()
    return p == r or r in p.parents


def verify_clean_source(
    repo_root: str | Path,
    *,
    authorized_source_sha: str,
    output_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    actual = _git(root, "rev-parse", "HEAD")
    if actual != authorized_source_sha:
        raise SourceStateError(f"source HEAD mismatch: expected {authorized_source_sha}, got {actual}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SourceStateError(f"repository working tree is not clean: {status[:2000]}")
    tree_sha = _git(root, "rev-parse", "HEAD^{tree}")
    for candidate in output_roots or []:
        if candidate and is_within(candidate, root):
            raise SourceStateError(f"mutable output/evidence root must be outside repository tree: {candidate}")
    return {
        "status": "PASS",
        "authorized_source_sha": authorized_source_sha,
        "observed_source_sha": actual,
        "tree_sha": tree_sha,
        "tracked_and_untracked_status_clean": True,
        "output_roots_outside_repository": True,
    }
