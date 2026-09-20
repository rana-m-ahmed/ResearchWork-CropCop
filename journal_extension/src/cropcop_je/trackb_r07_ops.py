from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hashing import sha256_file, sha256_json
from .publication import audit_public_files, publish_to_github_branch


class TrackBOpsError(RuntimeError):
    pass


KAGGLE_OWNER_DEFAULT = "AUTO"


def historical_dataset_slug(owner: str) -> str:
    owner = str(owner).strip()
    if not owner or "/" in owner:
        raise TrackBOpsError(f"invalid Kaggle owner: {owner!r}")
    return f"{owner}/cropcop-trackb-historical-r07-v2"


def evidence_dataset_slug(owner: str) -> str:
    owner = str(owner).strip()
    if not owner or "/" in owner:
        raise TrackBOpsError(f"invalid Kaggle owner: {owner!r}")
    return f"{owner}/cropcop-trackb-r07-evidence"

SOURCE_DATASETS = {
    "final_v1": "ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1",
    "r07_s1": "sabahatabbas/sec-je-r07-cnxtt-context-s1-8904b100d223-a01",
    "r07_s2": "sabahatabbas/cropcop-r07-cnxtt-context-s2-abce1197-56023042",
    "r07_s3": "sabahatabbas/cropcop-r07-cnxtt-context-s3-f13ca687-56023042",
    "dino_bundle": "ranamuhammadahmed6/cropcop-secondary-g1-8904b100",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_kaggle_secret(name: str) -> str:
    # The operator notebook reads Kaggle secrets once, then passes them only through
    # the fresh scientific subprocess environment. Environment values therefore take
    # precedence and no secret value is persisted to evidence.
    value = (os.environ.get(name) or "").strip()
    if not value:
        try:
            from kaggle_secrets import UserSecretsClient
        except Exception as exc:
            raise TrackBOpsError(
                f"required secret {name} is not present in the environment and Kaggle UserSecretsClient is unavailable"
            ) from exc
        try:
            value = (UserSecretsClient().get_secret(name) or "").strip()
        except Exception as exc:
            raise TrackBOpsError(f"required Kaggle secret is unavailable: {name}") from exc
    if not value:
        raise TrackBOpsError(f"required Kaggle secret is empty: {name}")
    if any(ch.isspace() for ch in value):
        raise TrackBOpsError(f"Kaggle secret contains whitespace/newline: {name}")
    return value


def configure_runtime_secrets() -> dict[str, bool]:
    kaggle = load_kaggle_secret("KAGGLE_API_TOKEN")
    github = load_kaggle_secret("CROPCOP_GITHUB_TOKEN")
    os.environ["KAGGLE_API_TOKEN"] = kaggle
    os.environ["CROPCOP_GITHUB_TOKEN"] = github
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    return {"KAGGLE_API_TOKEN": True, "CROPCOP_GITHUB_TOKEN": True}


def _github_askpass_environment(token: str, askpass: Path) -> dict[str, str]:
    token = token.strip()
    if not token:
        raise TrackBOpsError("CROPCOP_GITHUB_TOKEN is empty")
    if any(ch.isspace() for ch in token):
        raise TrackBOpsError("CROPCOP_GITHUB_TOKEN contains whitespace/newlines")
    env = dict(os.environ)
    env["CROPCOP_GITHUB_TOKEN"] = token
    env["CROPCOP_GIT_USERNAME"] = (
        os.environ.get("CROPCOP_GIT_USERNAME", "").strip() or "x-access-token"
    )
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_ASKPASS_REQUIRE"] = "force"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "credential.helper"
    env["GIT_CONFIG_VALUE_0"] = ""
    return env


def _classify_github_push_failure(detail: str) -> str:
    low = detail.lower()
    if any(marker in low for marker in (
        "authentication failed",
        "invalid username or token",
        "bad credentials",
        "could not read username",
    )):
        return (
            "GitHub credential is invalid, expired, revoked, or not a raw PAT. "
            "Replace Kaggle Secret CROPCOP_GITHUB_TOKEN with a valid GitHub PAT."
        )
    if any(marker in low for marker in (
        "permission to",
        "write access to repository not granted",
        "403",
        "denied to",
    )):
        return (
            "GitHub credential authenticated but lacks repository write permission. "
            "For a fine-grained PAT, grant repository access to ResearchWork-CropCop "
            "with Contents: Read and write."
        )
    if any(marker in low for marker in (
        "could not resolve host",
        "connection timed out",
        "failed to connect",
        "connection reset",
        "temporary failure",
        "network is unreachable",
    )):
        return "GitHub network transport failed; this looks transient rather than a token-scope failure."
    return "GitHub write preflight failed for an unclassified Git transport reason."


def verify_github_repository_push_access(
    repo_dir: str | Path,
    *,
    source_git_sha: str | None = None,
) -> dict[str, str | bool]:
    token = (os.environ.get("CROPCOP_GITHUB_TOKEN") or "").strip()
    if not token:
        raise TrackBOpsError("CROPCOP_GITHUB_TOKEN is not configured")
    repo_dir = Path(repo_dir).resolve()
    if not (repo_dir / ".git").exists():
        raise TrackBOpsError(f"GitHub push preflight requires a Git checkout: {repo_dir}")

    observed_head = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if source_git_sha and observed_head != str(source_git_sha).strip():
        raise TrackBOpsError(
            f"GitHub push preflight source mismatch: expected={source_git_sha}, observed={observed_head}"
        )

    remote_url = subprocess.check_output(
        ["git", "-C", str(repo_dir), "remote", "get-url", "origin"],
        text=True,
    ).strip()
    parsed = urllib.parse.urlparse(remote_url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise TrackBOpsError(
            f"GitHub push preflight requires an HTTPS github.com origin; observed {remote_url!r}"
        )

    with tempfile.TemporaryDirectory() as td:
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
        env = _github_askpass_environment(token, askpass)

        probe_ref = f"refs/heads/run-evidence/auth-probe-{observed_head[:12]}"
        args = [
            "git", "-C", str(repo_dir), "push", "--dry-run",
            "origin", f"HEAD:{probe_ref}",
        ]
        proc = subprocess.run(
            args,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if proc.returncode != 0:
            detail = redact(((proc.stderr or "") + "\n" + (proc.stdout or "")).strip())
            classification = _classify_github_push_failure(detail)
            if len(detail) > 1800:
                detail = detail[-1800:]
            raise TrackBOpsError(
                f"{classification} git push --dry-run diagnostic: {detail or '<no diagnostic>'}"
            )

    return {
        "repository_origin": remote_url,
        "authenticated": True,
        "push": True,
        "preflight": "GIT_PUSH_DRY_RUN_NO_REMOTE_REF_CREATED",
        "source_git_sha": observed_head,
    }


def _secret_values() -> list[str]:
    return [
        value for value in (
            os.environ.get("KAGGLE_API_TOKEN", ""),
            os.environ.get("CROPCOP_GITHUB_TOKEN", ""),
            os.environ.get("GITHUB_TOKEN", ""),
        ) if value
    ]


def redact(text: str) -> str:
    safe = text or ""
    for secret in _secret_values():
        safe = safe.replace(secret, "<redacted>")
    safe = re.sub(r"(KAGGLE_API_TOKEN|CROPCOP_GITHUB_TOKEN|GITHUB_TOKEN)\s*[=:]\s*\S+", r"\1=<redacted>", safe)
    return safe


def run_checked(args: list[str], *, cwd: str | Path | None = None, timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = subprocess.run(
        args,
        cwd=None if cwd is None else str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = redact((proc.stderr or proc.stdout or "").strip())
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise TrackBOpsError(f"command failed rc={proc.returncode}: {' '.join(args)}\n{detail}")
    return proc


def ensure_kaggle_cli() -> str:
    result = run_checked(["kaggle", "--version"], timeout=120)
    version = (result.stdout or result.stderr or "").strip()
    if not version:
        raise TrackBOpsError("Kaggle CLI version probe returned no output")
    return version


def _owners_from_csv(text: str) -> set[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    owners: set[str] = set()
    for line in lines[1:]:
        ref = line.split(",", 1)[0].strip().strip('"')
        if "/" in ref:
            owner = ref.split("/", 1)[0].strip()
            if owner:
                owners.add(owner)
    return owners


def detect_authenticated_kaggle_owner() -> str:
    # Prefer owned datasets because Track B needs dataset create/version permission.
    datasets = run_checked(["kaggle", "datasets", "list", "--mine", "-v"], timeout=180)
    owners = _owners_from_csv(datasets.stdout)
    if len(owners) == 1:
        return next(iter(owners))
    if len(owners) > 1:
        raise TrackBOpsError(
            f"authenticated Kaggle identity is ambiguous across owned datasets: {sorted(owners)}"
        )

    # Fresh accounts may own no datasets yet but will own the notebook currently
    # running Track B. The kernels surface gives us a second authenticated owner
    # signal without requiring any extra user input.
    kernels = run_checked(["kaggle", "kernels", "list", "--mine", "-v"], timeout=180)
    owners = _owners_from_csv(kernels.stdout)
    if len(owners) == 1:
        return next(iter(owners))
    raise TrackBOpsError(
        "could not uniquely infer the authenticated Kaggle owner from owned datasets "
        f"or notebooks; observed owners={sorted(owners)}. Supply --kaggle-owner explicitly."
    )


def verify_kaggle_source_access(source_datasets: dict[str, str]) -> dict[str, str]:
    verified: dict[str, str] = {}
    failures: dict[str, str] = {}
    for role, slug in source_datasets.items():
        proc = subprocess.run(
            ["kaggle", "datasets", "files", slug, "--page-size", "1"],
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if proc.returncode == 0:
            verified[role] = slug
        else:
            failures[role] = redact((proc.stderr or proc.stdout or "").strip())[-1200:]
    if failures:
        raise TrackBOpsError(
            "Kaggle token cannot access all frozen Track-B source datasets: "
            + json.dumps(failures, sort_keys=True)
        )
    return verified


def verify_authenticated_kaggle_owner(expected_owner: str) -> str:
    expected_owner = str(expected_owner).strip()
    observed = detect_authenticated_kaggle_owner()
    if expected_owner and expected_owner.upper() != "AUTO" and observed != expected_owner:
        raise TrackBOpsError(
            f"authenticated Kaggle owner mismatch: expected={expected_owner}, observed={observed}"
        )
    return observed


def download_kaggle_dataset(slug: str, destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    run_checked(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(destination), "--unzip", "-q"],
        timeout=7200,
    )
    # Some CLI versions retain the transport ZIP after --unzip. It is never an input authority.
    for path in destination.glob("*.zip"):
        try:
            path.unlink()
        except OSError:
            pass
    if not any(destination.iterdir()):
        raise TrackBOpsError(f"Kaggle dataset download produced an empty directory: {slug}")
    return destination


def kaggle_dataset_exists(slug: str) -> bool:
    proc = subprocess.run(
        ["kaggle", "datasets", "files", slug, "--page-size", "1"],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if proc.returncode == 0:
        return True
    combined = redact((proc.stdout or "") + "\n" + (proc.stderr or "")).lower()
    if any(marker in combined for marker in ("404", "not found", "dataset not found")):
        return False
    raise TrackBOpsError(f"could not determine Kaggle dataset existence for {slug}: {combined[-2000:]}")


def _metadata_slug(slug: str) -> tuple[str, str]:
    if "/" not in slug:
        raise TrackBOpsError(f"Kaggle dataset slug must be owner/dataset: {slug}")
    owner, dataset = slug.split("/", 1)
    if not owner or not dataset:
        raise TrackBOpsError(f"invalid Kaggle dataset slug: {slug}")
    return owner, dataset


def _local_kaggle_content_manifest(folder: Path) -> dict:
    ignored = {"dataset-metadata.json", "TRACKB_KAGGLE_CONTENT_MANIFEST.json"}
    files = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name in ignored:
            continue
        rel = path.relative_to(folder).as_posix()
        files.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not files:
        raise TrackBOpsError(f"private Kaggle publication contains no payload files: {folder}")
    digest = sha256_json(files)
    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "content_digest_sha256": digest,
        "files": files,
    }
    (folder / "TRACKB_KAGGLE_CONTENT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _kaggle_dataset_status(slug: str) -> dict:
    result = run_checked(
        ["kaggle", "datasets", "status", slug, "--format", "json"],
        timeout=180,
    )
    try:
        payload = json.loads(result.stdout)
    except Exception as exc:
        raise TrackBOpsError(
            f"Kaggle dataset status did not return valid JSON for {slug}: {redact(result.stdout)[-1000:]}"
        ) from exc
    if not isinstance(payload, dict):
        raise TrackBOpsError(f"Kaggle dataset status payload is not an object for {slug}")
    return payload


def _wait_kaggle_dataset_ready(slug: str, *, timeout_seconds: int = 900) -> dict:
    deadline = time.monotonic() + int(timeout_seconds)
    last = {}
    while time.monotonic() < deadline:
        last = _kaggle_dataset_status(slug)
        status = str(last.get("status", "")).strip().lower()
        if status == "ready":
            return last
        if status == "error":
            raise TrackBOpsError(f"Kaggle dataset entered error state: {slug}: {last}")
        time.sleep(5)
    raise TrackBOpsError(f"Kaggle dataset did not become ready within {timeout_seconds}s: {slug}: {last}")


def _download_kaggle_file(slug: str, relative_path: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "kaggle", "datasets", "download", slug,
            "-f", relative_path,
            "-p", str(destination),
            "--unzip", "-o", "-q",
        ],
        timeout=7200,
    )
    direct = destination / relative_path
    if direct.is_file():
        return direct
    matches = [p for p in destination.rglob(Path(relative_path).name) if p.is_file()]
    if len(matches) != 1:
        raise TrackBOpsError(
            f"Kaggle single-file round-trip could not resolve {relative_path!r} from {slug}: {matches}"
        )
    return matches[0]


def _read_remote_kaggle_content_manifest(slug: str) -> dict | None:
    if not kaggle_dataset_exists(slug):
        return None
    with tempfile.TemporaryDirectory() as td:
        try:
            path = _download_kaggle_file(
                slug,
                "TRACKB_KAGGLE_CONTENT_MANIFEST.json",
                Path(td),
            )
        except Exception:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TrackBOpsError(f"remote Kaggle content manifest is invalid JSON: {slug}") from exc
    if not isinstance(payload, dict) or len(str(payload.get("content_digest_sha256", ""))) != 64:
        raise TrackBOpsError(f"remote Kaggle content manifest is malformed: {slug}")
    return payload


def _verify_remote_kaggle_content(
    slug: str,
    local_manifest: dict,
    *,
    full_roundtrip: bool,
) -> dict:
    remote = _read_remote_kaggle_content_manifest(slug)
    if remote is None:
        raise TrackBOpsError(f"remote Kaggle content manifest missing after publication: {slug}")
    if remote.get("content_digest_sha256") != local_manifest.get("content_digest_sha256"):
        raise TrackBOpsError(
            f"remote Kaggle content digest mismatch: local={local_manifest.get('content_digest_sha256')} "
            f"remote={remote.get('content_digest_sha256')}"
        )
    if remote.get("files") != local_manifest.get("files"):
        raise TrackBOpsError("remote Kaggle content manifest file ledger differs from local publication ledger")
    verified_files = 0
    if full_roundtrip:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for row in local_manifest["files"]:
                rel = str(row["path"])
                path = _download_kaggle_file(slug, rel, root / f"f{verified_files:05d}")
                if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
                    raise TrackBOpsError(f"remote Kaggle round-trip bytes differ for {slug}/{rel}")
                verified_files += 1
    status = _wait_kaggle_dataset_ready(slug)
    version = status.get("current_version_number")
    if version is None:
        version = status.get("currentVersionNumber")
    return {
        "status": "PASS",
        "content_digest_sha256": local_manifest["content_digest_sha256"],
        "remote_manifest_match": True,
        "full_roundtrip": bool(full_roundtrip),
        "roundtrip_verified_file_count": verified_files,
        "current_version_number": version,
        "kaggle_status": status,
    }


def publish_private_kaggle_dataset(
    *,
    folder: str | Path,
    slug: str,
    title: str,
    version_message: str,
    license_name: str = "other",
    full_roundtrip: bool = False,
) -> dict[str, object]:
    folder = Path(folder).resolve()
    if not folder.is_dir() or not any(folder.iterdir()):
        raise TrackBOpsError(f"private Kaggle publication folder is empty: {folder}")
    owner, dataset = _metadata_slug(slug)
    metadata_path = folder / "dataset-metadata.json"
    metadata = {
        "title": title,
        "id": f"{owner}/{dataset}",
        "licenses": [{"name": license_name}],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    local_manifest = _local_kaggle_content_manifest(folder)
    local_digest = str(local_manifest["content_digest_sha256"])

    existed_initially = kaggle_dataset_exists(slug)
    errors: list[str] = []
    for attempt, delay in enumerate((0, 5, 15, 30), start=1):
        if delay:
            time.sleep(delay)
        try:
            existing = _read_remote_kaggle_content_manifest(slug)
            if existing and existing.get("content_digest_sha256") == local_digest:
                verification = _verify_remote_kaggle_content(
                    slug, local_manifest, full_roundtrip=full_roundtrip
                )
                return {
                    "slug": slug,
                    "action": "reuse",
                    "created": False,
                    "versioned": False,
                    "reused_identical_remote": True,
                    "attempts": attempt,
                    **verification,
                }

            exists_now = kaggle_dataset_exists(slug)
            if exists_now:
                run_checked(
                    [
                        "kaggle", "datasets", "version",
                        "-p", str(folder),
                        "-m", version_message,
                        "-q", "-r", "zip",
                    ],
                    timeout=7200,
                )
                action = "version"
            else:
                run_checked(
                    [
                        "kaggle", "datasets", "create",
                        "-p", str(folder),
                        "-q", "-r", "zip",
                    ],
                    timeout=7200,
                )
                action = "create"

            _wait_kaggle_dataset_ready(slug)
            verification = _verify_remote_kaggle_content(
                slug, local_manifest, full_roundtrip=full_roundtrip
            )
            return {
                "slug": slug,
                "action": action,
                "created": action == "create" and not existed_initially,
                "versioned": action == "version",
                "reused_identical_remote": False,
                "attempts": attempt,
                **verification,
            }
        except Exception as exc:
            message = redact(str(exc))
            errors.append(f"attempt={attempt} {type(exc).__name__}: {message[-1200:]}")
            if attempt < 4:
                print(
                    f"Private Kaggle publication attempt {attempt}/4 failed; retrying: "
                    f"{type(exc).__name__}: {message[-500:]}",
                    flush=True,
                )

    raise TrackBOpsError(
        "private Kaggle dataset publication failed after 4 attempts: "
        + " | ".join(errors[-4:])
    )



def _urlopen_json(url: str, *, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "CropCop-TrackB/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise TrackBOpsError(f"JSON object expected from {url}")
    return obj


def _checksum_matches(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    value = str(expected).strip().lower()
    if ":" in value:
        algorithm, digest = value.split(":", 1)
    else:
        digest = value
        algorithm = "sha256" if len(digest) == 64 else "md5" if len(digest) == 32 else ""
    if algorithm == "sha256":
        observed = sha256_file(path)
    elif algorithm == "md5":
        h = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        observed = h.hexdigest()
    else:
        raise TrackBOpsError(f"unsupported source checksum format: {expected!r}")
    return observed.lower() == digest.lower()


def _stream_download(
    url: str,
    destination: Path,
    *,
    timeout: int = 1800,
    expected_checksum: str | None = None,
    attempts: int = 4,
) -> dict[str, str | int | bool]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    errors: list[str] = []
    for attempt in range(1, int(attempts) + 1):
        partial.unlink(missing_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CropCop-TrackB/3.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, partial.open("wb") as fh:
                content_disposition = response.headers.get("Content-Disposition", "")
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    fh.write(block)
            if not partial.is_file() or partial.stat().st_size <= 0:
                raise TrackBOpsError("source download produced an empty file")
            if not _checksum_matches(partial, expected_checksum):
                raise TrackBOpsError(
                    f"source checksum mismatch for {destination.name}: expected={expected_checksum}"
                )
            os.replace(partial, destination)
            return {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "source_checksum": str(expected_checksum or ""),
                "source_checksum_verified": bool(expected_checksum),
                "content_disposition": content_disposition,
                "download_attempts": attempt,
            }
        except Exception as exc:
            partial.unlink(missing_ok=True)
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
            if attempt < int(attempts):
                time.sleep((2, 5, 15)[min(attempt - 1, 2)])
    raise TrackBOpsError(
        f"source download failed after {attempts} attempts: {url}: " + " | ".join(errors[-4:])
    )


def _safe_extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise TrackBOpsError(f"unsafe archive member: {info.filename}")
        zf.extractall(destination)


def _load_lineage_review(path: str | Path, *, role: str, doi: str, version: str) -> tuple[dict, str, str]:
    review_path = Path(path).resolve()
    if not review_path.is_file():
        raise TrackBOpsError(f"lineage review file missing: {review_path}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("review_id") != "TRACKB_EXTERNAL_LINEAGE_REVIEW_v1":
        raise TrackBOpsError("unexpected external-lineage review identity")
    row = (review.get("candidates") or {}).get(role)
    if not isinstance(row, dict):
        raise TrackBOpsError(f"lineage review lacks candidate role: {role}")
    if str(row.get("doi")) != doi or str(row.get("version")) != version:
        raise TrackBOpsError(f"lineage review source identity mismatch for {role}")
    status = str(row.get("lineage_review_status", ""))
    known = row.get("known_historical_contributor_relationship")
    if status not in {"PASS_NO_KNOWN_RELATIONSHIP", "RESIDUAL_UNCERTAINTY"} or not isinstance(known, bool):
        raise TrackBOpsError(f"lineage review conclusion invalid for {role}")
    return row, sha256_file(review_path), str(review.get("review_id"))


def acquire_irish_potato(destination: str | Path, *, lineage_review_path: str | Path) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    lineage_row, lineage_sha, lineage_id = _load_lineage_review(
        lineage_review_path,
        role="irish_potato",
        doi="10.5281/zenodo.8286529",
        version="01",
    )
    record = _urlopen_json("https://zenodo.org/api/records/8286529")
    metadata = record.get("metadata") or {}
    version = str(metadata.get("version", ""))
    if version != "01":
        raise TrackBOpsError(f"Zenodo Irish Potato version drift: expected 01, got {version!r}")
    files = record.get("files") or []
    expected = {"earlyblt.zip": 17772, "healthy.zip": 20438, "lateblt.zip": 20499}
    by_name = {str(row.get("key", "")): row for row in files}
    if set(expected).difference(by_name):
        raise TrackBOpsError(f"Zenodo record lacks required ZIP files: {sorted(set(expected).difference(by_name))}")
    data_root = destination / "data"
    transport = []
    observed_support = {}
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for name in ("earlyblt.zip", "healthy.zip", "lateblt.zip"):
        row = by_name[name]
        links = row.get("links") or {}
        url = links.get("self") or links.get("content")
        if not url:
            raise TrackBOpsError(f"Zenodo file has no download URL: {name}")
        archive = destination / "_transport" / name
        checksum = str(row.get("checksum") or "").strip()
        if not checksum:
            raise TrackBOpsError(f"Zenodo file lacks published checksum: {name}")
        receipt = _stream_download(str(url), archive, expected_checksum=checksum)
        class_name = name[:-4]
        extracted_root = destination / "_extracted" / class_name
        _safe_extract_zip(archive, extracted_root)
        images = sorted(
            path for path in extracted_root.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        expected_count = int(expected[name])
        if len(images) != expected_count:
            raise TrackBOpsError(
                f"Zenodo {name} image-count drift: expected {expected_count}, found {len(images)}"
            )
        class_root = data_root / class_name
        class_root.mkdir(parents=True, exist_ok=False)
        for src in images:
            raw_sha = sha256_file(src)
            member_rel = src.relative_to(extracted_root).as_posix()
            member_sha = hashlib.sha256(member_rel.encode("utf-8")).hexdigest()
            target = class_root / f"{raw_sha[:16]}_{member_sha[:12]}{src.suffix.lower()}"
            if target.exists():
                raise TrackBOpsError(f"canonical Irish Potato member collision: {target.name}")
            shutil.move(str(src), str(target))
        observed_support[class_name] = len(images)
        transport.append({"name": name, **receipt, "normalized_image_count": len(images)})
        archive.unlink(missing_ok=True)
        shutil.rmtree(extracted_root, ignore_errors=True)
    shutil.rmtree(destination / "_transport", ignore_errors=True)
    shutil.rmtree(destination / "_extracted", ignore_errors=True)
    licence = metadata.get("license") or {}
    licence_text = str(licence.get("id") or licence.get("title") or "").strip()
    if not licence_text:
        licence_text = "Public Zenodo record; exact license field not exposed in acquisition metadata"
    source = {
        "doi": "10.5281/zenodo.8286529",
        "version": "01",
        "source_url": "https://zenodo.org/records/8286529",
        "retrieved_at": utc_now(),
        "license_or_access_text": licence_text,
        "known_historical_contributor_relationship": bool(lineage_row["known_historical_contributor_relationship"]),
        "lineage_review_status": str(lineage_row["lineage_review_status"]),
        "lineage_review_id": lineage_id,
        "lineage_review_sha256": lineage_sha,
        "acquisition_transport": transport,
        "observed_class_support": observed_support,
        "zenodo_record_id": str(record.get("id", "")),
    }
    (destination / "SOURCE_METADATA.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return {"root": str(destination), "data_root": str(data_root), "source_metadata": source}


def _extract_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _extract_strings(v)
    elif isinstance(value, list):
        for item in value:
            yield from _extract_strings(item)


def _mendeley_download_links(html: str) -> list[str]:
    decoded = html.replace("\\u002F", "/").replace("\\/", "/")
    urls = set(re.findall(r'https?://[^"\'<>\\s]+', decoded))
    links = {
        url.rstrip(".,)")
        for url in urls
        if "mendeley.com" in url and ("file_downloaded" in url or "/public-files/" in url)
    }
    # Modern Mendeley pages often place download URLs or file IDs in __NEXT_DATA__.
    match = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html, flags=re.S | re.I)
    if match:
        try:
            obj = json.loads(match.group(1))
            for value in _extract_strings(obj):
                normalized = value.replace("\\u002F", "/").replace("\\/", "/")
                if normalized.startswith("http") and "mendeley.com" in normalized and (
                    "file_downloaded" in normalized or "/public-files/" in normalized
                ):
                    links.add(normalized)
        except Exception:
            pass
    return sorted(links)


def probe_external_sources() -> dict[str, object]:
    record = _urlopen_json("https://zenodo.org/api/records/8286529")
    metadata = record.get("metadata") or {}
    if str(metadata.get("version", "")) != "01":
        raise TrackBOpsError("Irish Potato source probe observed a non-01 Zenodo version")
    file_names = {str(row.get("key", "")) for row in (record.get("files") or [])}
    required_files = {"earlyblt.zip", "healthy.zip", "lateblt.zip"}
    if not required_files.issubset(file_names):
        raise TrackBOpsError(
            f"Irish Potato source probe is missing required archives: {sorted(required_files - file_names)}"
        )

    page_url = "https://data.mendeley.com/datasets/wkymf8bhcg/5"
    req = urllib.request.Request(
        page_url,
        headers={"User-Agent": "Mozilla/5.0 CropCop-TrackB/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise TrackBOpsError("GVLiD v5 Mendeley source page is unreachable") from exc
    links = _mendeley_download_links(html)
    if not links:
        raise TrackBOpsError(
            "GVLiD v5 source probe found no deterministic public-file download URL; "
            "do not start Track-B compute or substitute a mirror"
        )
    import hashlib
    return {
        "status": "PASS",
        "irish_potato": {
            "doi": "10.5281/zenodo.8286529",
            "version": "01",
            "required_archives_present": True,
        },
        "gvlid_v5": {
            "doi": "10.17632/wkymf8bhcg.5",
            "version": "5",
            "source_page_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "deterministic_download_link_count": len(links),
        },
    }


def _normalized_label(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    table = {
        "blackrot": "Black Rot",
        "esca": "Esca",
        "blackmeasles": "Esca",
        "healthy": "Healthy",
        "leafblight": "Leaf Blight",
        "isariopsisleafspot": "Leaf Blight",
    }
    return table.get(key)


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _parse_sha256_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$", line)
        if m:
            digest, name = m.group(1), m.group(2)
        else:
            m = re.match(r"^(.+?)\s*[:;,]\s*([0-9a-fA-F]{64})\s*$", line)
            if not m:
                continue
            name, digest = m.group(1), m.group(2)
        key = name.strip().replace("\\", "/").lstrip("./").lower()
        entries[key] = digest.lower()
    if not entries:
        raise TrackBOpsError(f"GVLiD checksum ledger contains no SHA-256 rows: {path}")
    return entries


def _verify_gvlid_checksum_ledger(extracted_root: Path, image_paths: list[Path]) -> dict[str, object]:
    ledgers = sorted(
        p for p in extracted_root.rglob("*")
        if p.is_file() and p.name.lower() == "checksums.txt"
    )
    if not ledgers:
        raise TrackBOpsError("GVLiD package lacks required docs/checksums.txt image-integrity ledger")
    parsed = [_parse_sha256_ledger(path) for path in ledgers]
    canonical = parsed[0]
    if any(rows != canonical for rows in parsed[1:]):
        raise TrackBOpsError("GVLiD package exposes conflicting checksum ledgers")

    verified = 0
    for image in image_paths:
        basename = image.name.lower()
        label = None
        for parent in image.parents:
            if parent == extracted_root:
                break
            label = _normalized_label(parent.name)
            if label:
                break
        candidates = []
        for key, digest in canonical.items():
            if key.endswith("/" + basename) or key == basename:
                ledger_label = _normalized_label(Path(key).parent.name)
                if label is None or ledger_label in {None, label}:
                    candidates.append((key, digest))
        if len(candidates) != 1:
            raise TrackBOpsError(
                f"GVLiD checksum ledger cannot uniquely bind image {image.name}: matches={len(candidates)}"
            )
        if sha256_file(image).lower() != candidates[0][1]:
            raise TrackBOpsError(f"GVLiD image SHA-256 mismatch against source ledger: {image}")
        verified += 1
    if verified != len(image_paths):
        raise TrackBOpsError("GVLiD checksum verification coverage mismatch")
    return {
        "ledger_files": [p.relative_to(extracted_root).as_posix() for p in ledgers],
        "ledger_entry_count": len(canonical),
        "verified_image_count": verified,
        "status": "PASS",
    }


def _normalize_gvlid_tree(extracted_root: Path, data_root: Path) -> tuple[dict[str, int], dict[str, object]]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    supports: dict[str, int] = {"Black Rot": 0, "Esca": 0, "Healthy": 0, "Leaf Blight": 0}
    images = sorted(
        path for path in extracted_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    integrity = _verify_gvlid_checksum_ledger(extracted_root, images)
    used: set[Path] = set()
    for path in images:
        label = None
        for parent in path.parents:
            if parent == extracted_root.parent:
                break
            label = _normalized_label(parent.name)
            if label:
                break
        if not label:
            raise TrackBOpsError(f"GVLiD image cannot be assigned to a frozen source class: {path}")
        if path in used:
            raise TrackBOpsError(f"GVLiD duplicate traversal path: {path}")
        used.add(path)
        raw_sha = sha256_file(path)
        member_rel = path.relative_to(extracted_root).as_posix()
        member_sha = hashlib.sha256(member_rel.encode("utf-8")).hexdigest()
        target = data_root / label / f"{raw_sha[:16]}_{member_sha[:12]}{path.suffix.lower()}"
        if target.exists():
            raise TrackBOpsError(f"canonical GVLiD member collision: {target.name}")
        _hardlink_or_copy(path, target)
        supports[label] += 1
    if sum(supports.values()) != 3477:
        raise TrackBOpsError(f"GVLiD v5 expected 3477 images, observed {supports} total={sum(supports.values())}")
    if any(value < 50 for value in supports.values()):
        raise TrackBOpsError(f"GVLiD class support below frozen minimum: {supports}")
    return supports, integrity


def acquire_gvlid_v5(destination: str | Path, *, lineage_review_path: str | Path) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    lineage_row, lineage_sha, lineage_id = _load_lineage_review(
        lineage_review_path,
        role="gvlid_v5",
        doi="10.17632/wkymf8bhcg.5",
        version="5",
    )
    page_url = "https://data.mendeley.com/datasets/wkymf8bhcg/5"
    req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0 CropCop-TrackB/2.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        html = response.read().decode("utf-8", errors="replace")
    page_sha = __import__("hashlib").sha256(html.encode("utf-8")).hexdigest()
    links = _mendeley_download_links(html)
    if not links:
        raise TrackBOpsError(
            "GVLiD v5 public Mendeley page did not expose deterministic public-file download URLs. "
            "Failing closed before candidate audit; do not substitute a Kaggle/HuggingFace mirror."
        )
    transport_dir = destination / "_transport"
    extracted = destination / "_extracted"
    receipts = []
    seen_transport_sha: set[str] = set()
    for url in links:
        provisional = transport_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:20] + ".download")
        receipt = _stream_download(url, provisional)
        transport_sha = str(receipt["sha256"])
        if transport_sha in seen_transport_sha:
            provisional.unlink(missing_ok=True)
            continue
        seen_transport_sha.add(transport_sha)
        target = transport_dir / f"{transport_sha[:20]}.zip"
        os.replace(provisional, target)
        receipts.append({"url": url, **{**receipt, "path": str(target)}})
        try:
            if zipfile.is_zipfile(target):
                _safe_extract_zip(target, extracted / f"part_{transport_sha[:20]}")
            else:
                raise TrackBOpsError(f"GVLiD public-file transport is not a ZIP archive: {target}")
        finally:
            target.unlink(missing_ok=True)
    data_root = destination / "data"
    supports, checksum_integrity = _normalize_gvlid_tree(extracted, data_root)
    shutil.rmtree(extracted, ignore_errors=True)
    source = {
        "doi": "10.17632/wkymf8bhcg.5",
        "version": "5",
        "source_url": page_url,
        "retrieved_at": utc_now(),
        "license_or_access_text": "CC BY 4.0",
        "known_historical_contributor_relationship": bool(lineage_row["known_historical_contributor_relationship"]),
        "lineage_review_status": str(lineage_row["lineage_review_status"]),
        "lineage_review_id": lineage_id,
        "lineage_review_sha256": lineage_sha,
        "source_page_sha256": page_sha,
        "observed_class_support": supports,
        "published_total_images": 3477,
        "published_class_count_table_status": "NOT_USED_AS_AUTHORITY_DUE_ONE_IMAGE_ARITHMETIC_DISCREPANCY",
        "source_checksum_integrity": checksum_integrity,
        "acquisition_transport": receipts,
    }
    (destination / "SOURCE_METADATA.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return {"root": str(destination), "data_root": str(data_root), "source_metadata": source}


def create_publication_staging(output_root: str | Path, destination: str | Path) -> list[Path]:
    output_root = Path(output_root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    selections: list[tuple[Path, str]] = []
    for name in (
        "r07_family_replay.json",
        "TRACKB_PREDICTION_FIREWALL.json",
        "TRACKB_FINAL_QA.json",
        "TRACKB_SCIENCE_MANIFEST.json",
        "TRACKB_FINAL_CLOSURE.json",
        "TRACKB_PACKAGE_MANIFEST.json",
    ):
        p = output_root / name
        if p.is_file():
            selections.append((p, name))
    for cid in ("gvlid_grape", "irish_potato"):
        for name in ("audit_summary.json", "seal.json"):
            p = output_root / "candidates" / cid / name
            if p.is_file():
                selections.append((p, f"{cid}__{name}"))
        for name in ("seed_metrics.json", "three_seed_summary.json", "bootstrap.json"):
            p = output_root / "analysis" / cid / name
            if p.is_file():
                selections.append((p, f"{cid}__{name}"))
    if not selections:
        raise TrackBOpsError("no public Track-B evidence was selected")
    staged = []
    for src, name in selections:
        dst = destination / name
        shutil.copy2(src, dst)
        staged.append(dst)
    index = {
        "schema_version": "2.0",
        "status": "PASS",
        "public_only": True,
        "raw_images_included": False,
        "model_weights_included": False,
        "row_level_predictions_included": False,
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(staged)
        ],
    }
    index_path = destination / "TRACKB_PUBLIC_EVIDENCE_INDEX.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    staged.append(index_path)
    return audit_public_files(staged)


def publish_public_trackb_evidence(
    *,
    repo_root: str | Path,
    source_git_sha: str,
    output_root: str | Path,
) -> dict[str, str]:
    output_root = Path(output_root).resolve()
    closure = json.loads((output_root / "TRACKB_FINAL_CLOSURE.json").read_text(encoding="utf-8"))
    qa = json.loads((output_root / "TRACKB_FINAL_QA.json").read_text(encoding="utf-8"))
    if closure.get("status") != "TRACK_B_CLOSED" or qa.get("status") != "PASS":
        raise TrackBOpsError("public evidence publication requires terminal Track-B closure and QA PASS")
    closure_sha = str(closure.get("closure_sha256", ""))
    science_sha = str(closure.get("trackb_science_sha256", ""))
    if len(closure_sha) != 64 or len(science_sha) != 64:
        raise TrackBOpsError("terminal closure lacks valid closure/science SHA-256 identity")
    run_id = f"TB3-FINAL-{science_sha[:16]}"
    with tempfile.TemporaryDirectory() as td:
        staged = create_publication_staging(output_root, Path(td) / "public")
        branch = publish_to_github_branch(
            repo_dir=repo_root,
            source_git_sha=source_git_sha,
            run_id=run_id,
            files=[str(p) for p in staged],
            destination_prefix="journal_extension/evidence/public/track_b",
        )
    return {"run_id": run_id, "branch": branch, "closure_sha256": closure_sha, "trackb_science_sha256": science_sha}


def prepare_private_evidence_folder(output_root: str | Path, destination: str | Path) -> Path:
    output_root = Path(output_root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in (
        "TRACKB_COMPLETE_EVIDENCE.zip",
        "TRACKB_PUBLIC_EVIDENCE.zip",
        "TRACKB_PACKAGE_MANIFEST.json",
        "TRACKB_FINAL_QA.json",
        "TRACKB_SCIENCE_MANIFEST.json",
        "TRACKB_FINAL_CLOSURE.json",
    ):
        src = output_root / name
        if not src.is_file():
            raise TrackBOpsError(f"terminal evidence artifact missing: {src}")
        shutil.copy2(src, destination / name)
    receipt = {
        "schema_version": "2.0",
        "status": "PASS",
        "restricted_archive": True,
        "raw_external_images_included": False,
        "model_weights_included": False,
        "created_at_utc": utc_now(),
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(destination.iterdir()) if p.is_file()
        ],
    }
    (destination / "TRACKB_RESTRICTED_ARCHIVE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return destination
