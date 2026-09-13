from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import tracka_v12_kaggle_operator_v3 as v3

SCIENCE_SHA = v3.SCIENCE_SHA
OperatorError = v3.OperatorError
_PUBLICATION_LOCK = threading.Lock()
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_DESTINATION_PREFIX = "journal_extension/evidence/public/runs"


def _publication_api(repo: Path):
    src = repo / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.publication import audit_public_files, publish_to_github_branch
    return audit_public_files, publish_to_github_branch


def _validated_inputs(repo: str | Path, run_id: str, files: list[str | Path]) -> tuple[Path, list[Path]]:
    repo = Path(repo).resolve()
    if not repo.is_dir():
        raise OperatorError(f"publication repo does not exist: {repo}")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise OperatorError(f"unsafe public evidence run_id: {run_id!r}")
    paths = [Path(path).resolve() for path in files]
    if not paths:
        raise OperatorError("public evidence publication requires at least one file")
    if len({path.name for path in paths}) != len(paths):
        raise OperatorError("public evidence publication requires unique basenames")
    audit_public_files, _ = _publication_api(repo)
    audit_public_files(paths)
    return repo, paths


def _fetch_remote_ref(repo: Path, run_id: str) -> str | None:
    branch = f"run-evidence/{run_id}"
    remote_ref = f"refs/remotes/origin/{branch}"
    cp = subprocess.run(
        [
            "git", "-C", str(repo), "fetch", "--quiet", "--force", "origin",
            f"refs/heads/{branch}:{remote_ref}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return remote_ref if cp.returncode == 0 else None


def _assert_source_ancestry(repo: Path, remote_ref: str) -> None:
    cp = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", SCIENCE_SHA, remote_ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if cp.returncode != 0:
        raise OperatorError(
            f"existing public evidence branch does not descend from frozen science SHA {SCIENCE_SHA}: {remote_ref}"
        )


def _remote_bytes(repo: Path, remote_ref: str, run_id: str, filename: str) -> bytes | None:
    repo_path = f"{_DESTINATION_PREFIX}/{run_id}/{filename}"
    cp = subprocess.run(
        ["git", "-C", str(repo), "show", f"{remote_ref}:{repo_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return cp.stdout if cp.returncode == 0 else None


def remote_evidence_matches(repo: str | Path, run_id: str, files: list[str | Path]) -> bool:
    repo, paths = _validated_inputs(repo, run_id, files)
    remote_ref = _fetch_remote_ref(repo, run_id)
    if remote_ref is None:
        return False
    _assert_source_ancestry(repo, remote_ref)
    for path in paths:
        remote = _remote_bytes(repo, remote_ref, run_id, path.name)
        if remote is None or remote != path.read_bytes():
            return False
    return True


def publish_public_files(
    repo: str | Path,
    run_id: str,
    files: list[str | Path],
    *,
    attempts: int = 3,
) -> str:
    """Publish allowlisted text evidence idempotently without changing frozen science code.

    Existing byte-identical evidence is a successful no-op. Differing evidence is updated through
    the frozen publisher. If a publish raises after the remote was actually updated, a post-failure
    byte comparison recognizes the completed operation. Existing evidence branches must descend
    from the frozen scientific source SHA.
    """
    if attempts < 1:
        raise OperatorError("publication attempts must be positive")
    repo, paths = _validated_inputs(repo, run_id, files)
    v3.load_github_token()
    _, frozen_publish = _publication_api(repo)
    branch = f"run-evidence/{run_id}"
    last_error: Exception | None = None

    with _PUBLICATION_LOCK:
        for attempt in range(1, attempts + 1):
            if remote_evidence_matches(repo, run_id, paths):
                print(f"Public evidence already canonical: {branch}")
                return branch
            try:
                result = frozen_publish(
                    repo_dir=repo,
                    source_git_sha=SCIENCE_SHA,
                    run_id=run_id,
                    files=[str(path) for path in paths],
                )
                if not remote_evidence_matches(repo, run_id, paths):
                    raise OperatorError(f"public evidence did not round-trip after publish: {branch}")
                return result
            except Exception as exc:
                last_error = exc
                try:
                    if remote_evidence_matches(repo, run_id, paths):
                        print(f"Public evidence became canonical despite client-side publish error: {branch}")
                        return branch
                except Exception as verification_exc:
                    last_error = verification_exc
                if attempt == attempts:
                    break
                delay = (5, 15, 30)[min(attempt - 1, 2)]
                print(
                    f"public evidence publication attempt {attempt}/{attempts} failed for {run_id}: "
                    f"{type(last_error).__name__}; retrying in {delay}s"
                )
                time.sleep(delay)

    assert last_error is not None
    raise last_error


def install_stage_publication_hooks() -> None:
    """Route all pre-science master stages through the idempotent publisher."""
    import master_g1a
    import master_g2a
    import master_control

    master_g1a.publish_public_files = publish_public_files
    master_g2a.publish_public_files = publish_public_files
    master_control.publish_public_files = publish_public_files
