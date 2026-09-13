from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from tracka_v12_kaggle_operator_v2 import *  # noqa: F401,F403

OPERATOR_SCHEMA_VERSION = "3.0"
MASTER_ACCOUNTS = ("K1", "K2", "K3")
MASTER_WAIT_POLL_SECONDS = 30
MASTER_WAIT_MAX_SECONDS = 30 * 60
CODE_ATTESTATION_FILE_SHA256 = "b5421c6c91b7699651ac13c7352b7be9cb64735802d4a7d07bfb5282749f9212"
LOCK_RUNTIME_ATTESTATION_FILE_SHA256 = "fdf20eeea650215b61ca4b9201e78196d73bb7a7258adb4d0bd383d413198087"
_PUBLICATION_LOCK = threading.Lock()


def assert_kaggle_batch() -> None:
    run_type = str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip()
    if run_type != "Batch":
        raise OperatorError(
            "Canonical Track-A execution must use Kaggle Save Version -> Save & Run All / Batch. "
            f"Observed KAGGLE_KERNEL_RUN_TYPE={run_type or '<missing>'}. Interactive runs are diagnostic only."
        )


def ensure_locked_stack(repo: str | Path) -> dict:
    """Reuse an already-exact environment; install the frozen stack only when verification fails."""
    try:
        return verify_locked_stack(repo)
    except OperatorError as first_error:
        print(f"Frozen stack verification requires repair: {first_error}")
        print("Installing exact repository training lock once, then re-verifying.")
        install_locked_stack(repo)
        return verify_locked_stack(repo)


def load_github_token() -> str:
    token = str(os.environ.get("CROPCOP_GITHUB_TOKEN", "") or "").strip()
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = str(UserSecretsClient().get_secret("CROPCOP_GITHUB_TOKEN") or "").strip()
        except Exception as exc:
            raise OperatorError(
                "GitHub evidence publication requires Kaggle secret CROPCOP_GITHUB_TOKEN."
            ) from exc
    if not token or any(ch.isspace() for ch in token):
        raise OperatorError("CROPCOP_GITHUB_TOKEN is empty or contains whitespace")
    if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
        raise OperatorError("CROPCOP_GITHUB_TOKEN must be stored without surrounding quotes")
    os.environ["CROPCOP_GITHUB_TOKEN"] = token
    return token


def _askpass_env(token: str) -> tuple[dict[str, str], tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory(dir="/kaggle/working")
    askpass = Path(td.name) / "askpass.py"
    askpass.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "p=(sys.argv[1] if len(sys.argv)>1 else '').lower()\n"
        "if 'username' in p:\n"
        "    print('x-access-token')\n"
        "elif 'password' in p:\n"
        "    print(os.environ['CROPCOP_GITHUB_TOKEN'])\n"
        "else:\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["CROPCOP_GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_ASKPASS_REQUIRE"] = "force"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env, td


def github_write_preflight(repo: str | Path) -> None:
    token = load_github_token()
    env, td = _askpass_env(token)
    try:
        probe = f"refs/heads/run-evidence/auth-probe-tracka-v12-{SCIENCE_SHA[:12]}"
        cp = run(
            ["git", "-C", str(Path(repo).resolve()), "push", "--dry-run", "origin", f"HEAD:{probe}"],
            env=env,
            capture=True,
            check=False,
        )
        if cp.returncode != 0:
            detail = (cp.stdout or "").replace(token, "<redacted>")[-2500:]
            raise OperatorError(
                "GitHub evidence-branch write preflight failed. The PAT must have repository "
                f"Contents read/write permission.\n{detail}"
            )
    finally:
        td.cleanup()


def public_run_id(kind: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in kind.upper()).strip("-")
    return f"TRACKA-V12-{safe}-{SCIENCE_SHA[:12]}"


def g2a_public_run_id(calibration_id: str) -> str:
    return public_run_id(f"G2A-{calibration_id}")


def g1a_handoff_run_id() -> str:
    return public_run_id("G1A-HANDOFF")


def control_public_run_id() -> str:
    return public_run_id("CONTROL")


def account_public_run_id(account_id: str) -> str:
    return public_run_id(f"ACCOUNT-{account_id}")


def publish_public_files(
    repo: str | Path,
    run_id: str,
    files: list[str | Path],
    *,
    attempts: int = 3,
) -> str:
    """Serialize parent-side Git publication and retry transient failures without exposing credentials to children."""
    if attempts < 1:
        raise OperatorError("publication attempts must be positive")
    load_github_token()
    src = Path(repo).resolve() / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.publication import publish_to_github_branch

    last_error: Exception | None = None
    with _PUBLICATION_LOCK:
        for attempt in range(1, attempts + 1):
            try:
                return publish_to_github_branch(
                    repo_dir=Path(repo).resolve(),
                    source_git_sha=SCIENCE_SHA,
                    run_id=run_id,
                    files=[str(Path(p).resolve()) for p in files],
                )
            except Exception as exc:
                last_error = exc
                if attempt == attempts:
                    break
                delay = (5, 15, 30)[min(attempt - 1, 2)]
                print(
                    f"public evidence publication attempt {attempt}/{attempts} failed for {run_id}: "
                    f"{type(exc).__name__}; retrying in {delay}s"
                )
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def _fetch_evidence_branch(repo: str | Path, run_id: str) -> str | None:
    branch = f"run-evidence/{run_id}"
    remote_ref = f"refs/remotes/origin/{branch}"
    cp = run(
        [
            "git", "-C", str(Path(repo).resolve()), "fetch", "--quiet", "--force", "origin",
            f"refs/heads/{branch}:{remote_ref}",
        ],
        capture=True,
        check=False,
    )
    return remote_ref if cp.returncode == 0 else None


def fetch_public_file(
    repo: str | Path,
    run_id: str,
    filename: str,
    destination: str | Path,
) -> Path | None:
    remote_ref = _fetch_evidence_branch(repo, run_id)
    if remote_ref is None:
        return None
    repo_path = f"journal_extension/evidence/public/runs/{run_id}/{filename}"
    cp = run(
        ["git", "-C", str(Path(repo).resolve()), "show", f"{remote_ref}:{repo_path}"],
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        return None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(cp.stdout or "", encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def fetch_public_bundle(
    repo: str | Path,
    run_id: str,
    filenames: list[str],
    destination_dir: str | Path,
) -> dict[str, Path] | None:
    remote_ref = _fetch_evidence_branch(repo, run_id)
    if remote_ref is None:
        return None
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for filename in filenames:
        repo_path = f"journal_extension/evidence/public/runs/{run_id}/{filename}"
        cp = run(
            ["git", "-C", str(Path(repo).resolve()), "show", f"{remote_ref}:{repo_path}"],
            capture=True,
            check=False,
        )
        if cp.returncode != 0:
            return None
        destination = destination_dir / filename
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        tmp.write_text(cp.stdout or "", encoding="utf-8")
        os.replace(tmp, destination)
        result[filename] = destination
    return result


def wait_for_public_file(
    repo: str | Path,
    run_id: str,
    filename: str,
    destination: str | Path,
    *,
    predicate=None,
    max_seconds: float = MASTER_WAIT_MAX_SECONDS,
    poll_seconds: float = MASTER_WAIT_POLL_SECONDS,
) -> Path:
    started = time.monotonic()
    while True:
        path = fetch_public_file(repo, run_id, filename, destination)
        if path is not None:
            if predicate is None or predicate(load_json(path)):
                return path
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError(f"timed out waiting for GitHub evidence {run_id}/{filename}")
        time.sleep(poll_seconds)


def attestations_dir() -> Path:
    return Path(__file__).resolve().parent / "attestations"


def verified_attestation_paths() -> tuple[Path, Path]:
    code = attestations_dir() / "tracka-v12-pre-science-code-attestation.json"
    lock = attestations_dir() / "tracka-v12-exact-head-lock-runtime-attestation.json"
    if sha256_file(code) != CODE_ATTESTATION_FILE_SHA256:
        raise OperatorError("packaged code-attestation bytes changed")
    if sha256_file(lock) != LOCK_RUNTIME_ATTESTATION_FILE_SHA256:
        raise OperatorError("packaged lock/runtime-attestation bytes changed")
    code_payload = load_json(code)
    lock_payload = load_json(lock)
    if code_payload.get("status") != "PASS" or code_payload.get("pull_request_head_sha") != SCIENCE_SHA:
        raise OperatorError("packaged code attestation is not the frozen exact-head PASS")
    if lock_payload.get("status") != "PASS" or lock_payload.get("source_git_commit") != SCIENCE_SHA:
        raise OperatorError("packaged lock/runtime attestation is not the frozen exact-head PASS")
    return code, lock


def validate_g1a_bundle_with_science(repo: str | Path, bundle: str | Path) -> dict:
    src = Path(repo).resolve() / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.tracka_v12_runtime import load_and_validate_g1a_bundle

    payload, errors = load_and_validate_g1a_bundle(bundle, expected_source_sha=SCIENCE_SHA)
    if errors:
        raise OperatorError("G1A bundle validation failed: " + "; ".join(errors))
    if payload.get("status") != "PASS" or payload.get("science_authorized") is not False:
        raise OperatorError("G1A bundle is not canonical PASS/non-authorizing")
    return payload


def shared_g1a_locator(username: str) -> str:
    return f"{username}/{kaggle_safe_dataset_slug('cropcop-g1a', 'shared-canonical')}"


def kaggle_dataset_status(locator: str, *, env: dict[str, str]) -> str:
    cp = run(
        ["kaggle", "datasets", "status", locator, "--format", "json"],
        env=env,
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        return ""
    raw = (cp.stdout or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip().lower()
    if isinstance(payload, dict):
        for key in ("status", "currentVersionStatus", "current_version_status"):
            if key in payload:
                return str(payload[key]).strip().lower()
    return ""


def wait_kaggle_dataset_ready(
    locator: str,
    *,
    env: dict[str, str],
    max_seconds: float = MASTER_WAIT_MAX_SECONDS,
    poll_seconds: float = 20.0,
) -> None:
    ready = {"ready", "complete", "completed"}
    failed = {"error", "failed", "failure"}
    started = time.monotonic()
    while True:
        status = kaggle_dataset_status(locator, env=env)
        if status in failed:
            raise OperatorError(f"Kaggle dataset entered failure state: {locator} status={status}")
        if status in ready and kaggle_dataset_exists(locator, env=env):
            return
        if not status and kaggle_dataset_exists(locator, env=env):
            return
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError(f"timed out waiting for Kaggle dataset readiness: {locator}")
        time.sleep(poll_seconds)


def ensure_private_dataset(
    locator: str,
    *,
    title: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Create a private Kaggle dataset if absent and wait for authoritative readability."""
    env = dict(os.environ if env is None else env)
    owner, sep, slug = locator.partition("/")
    if not sep or not owner or not slug:
        raise OperatorError(f"invalid Kaggle locator: {locator}")
    if kaggle_dataset_exists(locator, env=env):
        return
    with tempfile.TemporaryDirectory(dir="/kaggle/working") as td:
        root = Path(td)
        write_json(root / "dataset-metadata.json", {
            "title": title or slug,
            "id": locator,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        })
        (root / "README.txt").write_text(
            "Private CropCop operational durability/handoff dataset. Do not make public.\n",
            encoding="utf-8",
        )
        cp = run(
            ["kaggle", "datasets", "create", "-p", str(root), "-q"],
            env=env,
            capture=True,
            check=False,
        )
        if cp.returncode != 0 and not kaggle_dataset_exists(locator, env=env):
            raise OperatorError(
                f"private Kaggle dataset creation failed for {locator}:\n{(cp.stdout or '')[-3000:]}"
            )
    wait_kaggle_dataset_ready(locator, env=env)


def download_and_validate_g1a_dataset(
    repo: str | Path,
    locator: str,
    destination: str | Path,
    *,
    env: dict[str, str],
) -> tuple[Path, dict]:
    root = download_private_dataset(locator, destination, env=env)
    bundle = discover_g1a_bundle(root)
    payload = validate_g1a_bundle_with_science(repo, bundle)
    return bundle, payload


def validate_g2a_summary(repo: str | Path, summary_path: str | Path, *, expected_g1a_sha: str) -> dict:
    src = Path(repo).resolve() / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.tracka_v12_g2a_durability import validate_calibration_durability
    from cropcop_je.tracka_v12_g2a_v122 import validate_calibration_summary_v122

    payload = load_json(summary_path)
    errors = validate_calibration_summary_v122(payload) + validate_calibration_durability(payload)
    if payload.get("source_git_commit") != SCIENCE_SHA:
        errors.append("source_git_commit mismatch")
    if payload.get("g1a_seal_sha256") != expected_g1a_sha:
        errors.append("g1a_seal_sha256 mismatch")
    if errors:
        raise OperatorError(
            f"G2A summary {payload.get('calibration_id')} invalid: " + "; ".join(errors)
        )
    return payload


def operator_runtime_head() -> str:
    repo = Path(__file__).resolve().parents[3]
    head = run(["git", "rev-parse", "HEAD"], cwd=repo, capture=True).stdout.strip()
    dirty = run(["git", "status", "--porcelain"], cwd=repo, capture=True).stdout.strip()
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head.lower()):
        raise OperatorError(f"operator runtime HEAD is invalid: {head!r}")
    if dirty:
        raise OperatorError(f"operator runtime checkout is dirty:\n{dirty}")
    return head


def adopt_g1a_if_present(
    repo: str | Path,
    locator: str,
    destination: str | Path,
    *,
    env: dict[str, str],
) -> tuple[Path, dict] | None:
    """Adopt an existing sealed G1A; an empty placeholder is recoverable, a stale seal is not."""
    wait_kaggle_dataset_ready(locator, env=env, max_seconds=120)
    root = download_private_dataset(locator, destination, env=env)
    seals = sorted(root.rglob("TRACKA_V12_G1A_SEAL.json"))
    if not seals:
        return None
    bundle = discover_g1a_bundle(root)
    payload = validate_g1a_bundle_with_science(repo, bundle)
    return bundle, payload


def technical_rollover_only(summary: dict) -> bool:
    if summary.get("status") != "ATTENTION_REQUIRED" or summary.get("science_complete") is not False:
        return False
    if summary.get("worker_errors"):
        return False
    slot_results = summary.get("slot_results")
    if not isinstance(slot_results, dict) or not slot_results:
        return False
    saw_continuation = False
    for rows in slot_results.values():
        if not isinstance(rows, list):
            return False
        for row in rows:
            if not isinstance(row, dict):
                return False
            if row.get("return_code") not in {0, None}:
                return False
            if row.get("run_status") == "FAIL" or row.get("slot_quarantined") is True:
                return False
            continuation = row.get("continuation_required") is True or row.get("session_rollover_required") is True
            if row.get("status") == "NOT_STARTED_SESSION_BUDGET":
                continuation = True
            saw_continuation = saw_continuation or continuation
    return saw_continuation
