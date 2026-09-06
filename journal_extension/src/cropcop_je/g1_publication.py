from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping

from .atomic_io import atomic_write_json
from .g1 import validate_g1_seal_object
from .g1_inputs import validate_private_slug
from .g1_package import (
    G1Package,
    EXPANDED_PACKAGE_DIR,
    G1PackageError,
    MANIFEST_NAME,
    PACKAGE_NAME,
    create_g1_package,
    locate_g1_package,
    reconstruct_expanded_g1_package,
    safe_extract_g1_package,
    validate_package_manifest,
)
from .hashing import sha256_file

STATUS_READY = {"ready", "complete", "completed"}
STATUS_FAILED = {"error", "failed", "failure"}
REQUIRED_REMOTE_FILES = (PACKAGE_NAME, MANIFEST_NAME)


class G1PublicationError(RuntimeError):
    pass


class G1RemoteMismatch(G1PublicationError):
    """The selected remote version exists but is not the expected package."""


def _required(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name, "")).strip()
    if not value:
        raise G1PublicationError(f"required Kaggle publication credential missing: {name}")
    return value


def _redact(text: str, env: Mapping[str, str]) -> str:
    safe = str(text or "")
    for name in ("KAGGLE_KEY", "CROPCOP_GITHUB_TOKEN", "GITHUB_TOKEN"):
        secret = str(env.get(name, "") or "")
        if secret:
            safe = safe.replace(secret, "<redacted>")
    return safe


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _metadata(api, slug: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(api.dataset_metadata(slug, td))
        payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("info") or payload


def _mine_refs(api, dataset_slug: str) -> set[str]:
    response = api.dataset_list_with_response(
        mine=True,
        search=dataset_slug,
        page_size=100,
    )
    datasets = getattr(response, "datasets", None) or []
    return {
        str(getattr(row, "ref", "") or "")
        for row in datasets
        if getattr(row, "ref", None)
    }


def _decode_status(raw) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    value = str(raw or "").strip()
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {"status": value}
    return payload if isinstance(payload, dict) else {"status": str(payload)}


def _version_number(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise G1PublicationError(f"invalid Kaggle current_version_number: {value!r}") from exc


def _status_snapshot(api, slug: str) -> dict:
    payload = _decode_status(api.dataset_status(slug, format="json"))
    return {
        "status": str(payload.get("status", "") or "").strip().lower(),
        "current_version_number": _version_number(payload.get("current_version_number")),
    }


def _validate_private_metadata(metadata: dict, slug: str) -> None:
    if metadata.get("isPrivate") is not True:
        raise G1PublicationError(
            "G1 target is not authoritatively reported private by Kaggle metadata"
        )
    meta_id = str(metadata.get("id") or metadata.get("ref") or "")
    if meta_id and meta_id.casefold() != slug.casefold():
        raise G1PublicationError(
            f"Kaggle metadata identity mismatch: {meta_id!r} != {slug!r}"
        )


def _target_probe(api, slug: str) -> dict:
    _, dataset_slug = slug.split("/", 1)
    state = {
        "metadata": None,
        "metadata_error": None,
        "mine_refs": set(),
        "mine_error": None,
        "status": None,
        "status_error": None,
    }
    try:
        state["metadata"] = _metadata(api, slug)
        _validate_private_metadata(state["metadata"], slug)
    except G1PublicationError:
        raise
    except Exception as exc:
        state["metadata_error"] = f"{type(exc).__name__}: {exc}"
    try:
        state["mine_refs"] = _mine_refs(api, dataset_slug)
    except Exception as exc:
        state["mine_error"] = f"{type(exc).__name__}: {exc}"
    try:
        state["status"] = _status_snapshot(api, slug)
    except Exception as exc:
        state["status_error"] = f"{type(exc).__name__}: {exc}"
    return state


def _target_summary(state: dict, slug: str) -> dict:
    refs = {str(ref).casefold() for ref in state.get("mine_refs", set())}
    status = state.get("status") or {}
    return {
        "metadata_visible": state.get("metadata") is not None,
        "metadata_error": state.get("metadata_error"),
        "mine_membership": slug.casefold() in refs,
        "mine_error": state.get("mine_error"),
        "dataset_status": status.get("status"),
        "current_version_number": status.get("current_version_number"),
        "status_error": state.get("status_error"),
    }


def _target_ready(state: dict, slug: str) -> bool:
    metadata = state.get("metadata")
    if metadata is None:
        return False
    _validate_private_metadata(metadata, slug)
    refs = {str(ref).casefold() for ref in state.get("mine_refs", set())}
    if slug.casefold() not in refs:
        return False
    status = state.get("status") or {}
    name = str(status.get("status", "") or "").strip().lower()
    if name in STATUS_FAILED:
        raise G1PublicationError(
            f"Kaggle G1 private-target processing failed with status={name!r}"
        )
    version = _version_number(status.get("current_version_number"))
    return name in STATUS_READY and version is not None and version >= 1


def _owner_preflight(slug: str, env: Mapping[str, str]) -> tuple[str, str]:
    slug = validate_private_slug(slug)
    username = _required(env, "KAGGLE_USERNAME")
    _required(env, "KAGGLE_KEY")
    owner, dataset_slug = slug.split("/", 1)
    if owner.casefold() != username.casefold():
        raise G1PublicationError(
            f"G1 private target owner mismatch: slug owner {owner!r} != "
            f"authenticated KAGGLE_USERNAME {username!r}"
        )
    return username, dataset_slug


def preflight_private_target(
    slug: str,
    *,
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[], object] = _api,
) -> dict:
    env = os.environ if env is None else env
    username, _ = _owner_preflight(slug, env)
    api = api_factory()
    state = _target_probe(api, slug)
    if not _target_ready(state, slug):
        raise G1PublicationError(
            "G1 private target is not fully settled: "
            + json.dumps(_target_summary(state, slug), sort_keys=True)
        )
    summary = _target_summary(state, slug)
    return {
        "schema_version": "2.0",
        "status": "PASS",
        "dataset_slug": slug,
        "authenticated_username": username,
        "owner_match": True,
        "mine_membership": True,
        "authoritative_is_private": True,
        "dataset_status": summary["dataset_status"],
        "current_version_number": summary["current_version_number"],
    }


def readiness_private_target_probe(
    slug: str,
    *,
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[], object] = _api,
) -> dict:
    """Readiness performs no target mutation and may return DEFERRED."""
    env = os.environ if env is None else env
    username, _ = _owner_preflight(slug, env)
    api = api_factory()
    state = _target_probe(api, slug)
    if _target_ready(state, slug):
        result = _target_summary(state, slug)
        result.update(
            {
                "schema_version": "2.0",
                "status": "PASS",
                "dataset_slug": slug,
                "authenticated_username": username,
                "authoritative_is_private": True,
            }
        )
        return result
    if state.get("mine_error"):
        raise G1PublicationError(
            "Kaggle authenticated 'mine' listing unavailable during readiness: "
            + str(state["mine_error"])
        )
    result = _target_summary(state, slug)
    result.update(
        {
            "schema_version": "2.0",
            "status": "DEFERRED",
            "dataset_slug": slug,
            "authenticated_username": username,
            "authoritative_is_private": None,
            "reason": (
                "private target absent or not fully settled; "
                "readiness performed no mutation"
            ),
        }
    )
    return result


def create_private_target_if_missing(
    slug: str,
    *,
    env: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> subprocess.CompletedProcess:
    env = os.environ if env is None else env
    _, dataset_slug = _owner_preflight(slug, env)
    if str(env.get("CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET", "")).strip() != "1":
        raise G1PublicationError(
            "private target auto-creation requires "
            "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1"
        )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "README.txt").write_text(
            "Private CropCop G1 transport target. "
            "Real sealed package versions are uploaded separately.\n",
            encoding="utf-8",
        )
        (root / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": dataset_slug,
                    "id": slug,
                    "licenses": [{"name": "other"}],
                    "isPrivate": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return run(
            ["kaggle", "datasets", "create", "-p", str(root), "-q", "-r", "skip"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )


def wait_until_ready(
    api,
    slug: str,
    *,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    deadline = clock() + timeout_seconds
    last = None
    while clock() < deadline:
        last = _status_snapshot(api, slug)
        status = str(last.get("status", "")).lower()
        if status in STATUS_READY:
            return last
        if status in STATUS_FAILED:
            raise G1PublicationError(f"Kaggle dataset processing failed: {last}")
        sleep_fn(poll_interval_seconds)
    raise G1PublicationError(
        f"Kaggle dataset did not become ready before timeout; last={last}"
    )


def wait_until_target_settled(
    api,
    slug: str,
    *,
    timeout_seconds: float = 600.0,
    poll_interval_seconds: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    deadline = clock() + timeout_seconds
    last = None
    attempt = 0
    while clock() < deadline:
        attempt += 1
        last = _target_probe(api, slug)
        if _target_ready(last, slug):
            summary = _target_summary(last, slug)
            summary["attempts"] = attempt
            return summary
        sleep_fn(poll_interval_seconds)
    raise G1PublicationError(
        "G1 private target did not become metadata-visible, mine-listed, and ready; "
        "last=" + json.dumps(_target_summary(last or {}, slug), sort_keys=True)
    )


def ensure_private_target(
    slug: str,
    *,
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[], object] = _api,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    settle_timeout_seconds: float = 600.0,
) -> dict:
    env = os.environ if env is None else env
    username, _ = _owner_preflight(slug, env)
    policy = str(env.get("CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET", "")).strip()
    if policy not in {"0", "1"}:
        raise G1PublicationError(
            "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET must be explicitly 0 or 1"
        )
    api = api_factory()
    initial = _target_probe(api, slug)
    if _target_ready(initial, slug):
        result = _target_summary(initial, slug)
        result.update(
            {
                "schema_version": "2.0",
                "status": "PASS",
                "dataset_slug": slug,
                "authenticated_username": username,
                "authoritative_is_private": True,
                "created_this_run": False,
                "creation_command_nonzero_but_target_settled": False,
            }
        )
        return result

    refs = {str(ref).casefold() for ref in initial.get("mine_refs", set())}
    existence_signal = (
        initial.get("metadata") is not None
        or initial.get("status") is not None
        or slug.casefold() in refs
    )
    if existence_signal:
        settled = wait_until_target_settled(
            api, slug, timeout_seconds=settle_timeout_seconds
        )
        settled.update(
            {
                "schema_version": "2.0",
                "status": "PASS",
                "dataset_slug": slug,
                "authenticated_username": username,
                "authoritative_is_private": True,
                "created_this_run": False,
                "creation_command_nonzero_but_target_settled": False,
            }
        )
        return settled
    if initial.get("mine_error"):
        raise G1PublicationError(
            "authenticated 'mine' listing unavailable; refusing target creation "
            "because ownership cannot be proven: " + str(initial["mine_error"])
        )
    if policy != "1":
        raise G1PublicationError(
            "G1 private target is absent/inaccessible and "
            "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0"
        )

    create = create_private_target_if_missing(slug, env=env, run=run)
    if create.returncode != 0:
        try:
            settled = wait_until_target_settled(
                api, slug, timeout_seconds=60.0
            )
        except Exception as settle_exc:
            detail = _redact(
                (getattr(create, "stderr", "") or "").strip()
                or (getattr(create, "stdout", "") or "").strip(),
                env,
            )
            raise G1PublicationError(
                "Kaggle private-target create command failed and target did not "
                f"settle; rc={create.returncode}; "
                f"detail={detail[-1600:] or '<no diagnostic>'}"
            ) from settle_exc
        settled.update(
            {
                "schema_version": "2.0",
                "status": "PASS",
                "dataset_slug": slug,
                "authenticated_username": username,
                "authoritative_is_private": True,
                "created_this_run": False,
                "creation_command_nonzero_but_target_settled": True,
            }
        )
        return settled

    settled = wait_until_target_settled(
        api, slug, timeout_seconds=settle_timeout_seconds
    )
    settled.update(
        {
            "schema_version": "2.0",
            "status": "PASS",
            "dataset_slug": slug,
            "authenticated_username": username,
            "authoritative_is_private": True,
            "created_this_run": True,
            "creation_command_nonzero_but_target_settled": False,
        }
    )
    return settled


def validate_local_g1_bundle(bundle_dir: str | Path) -> dict:
    bundle = Path(bundle_dir).resolve()
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file():
        raise G1PublicationError("G1_MODEL_IDENTITY_SEAL.json is missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    errors = validate_g1_seal_object(seal)
    if errors:
        raise G1PublicationError(
            "existing G1 bundle seal is invalid: " + "; ".join(errors)
        )
    for key in ("S1", "S2", "S3"):
        row = seal["pair_initializations"][key]
        path = bundle / "private" / row["basename"]
        if (
            not path.is_file()
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != int(row["bytes"])
        ):
            raise G1PublicationError(
                f"existing G1 pair artifact missing/corrupt: {key}"
            )
    for section, basename in (
        ("student", seal["student"]["pretrained"]["basename"]),
        ("teacher", seal["teacher"]["artifact_basename"]),
    ):
        path = bundle / "private" / basename
        expected_sha = (
            seal["student"]["pretrained"]["sha256"]
            if section == "student"
            else seal["teacher"]["checkpoint_sha256"]
        )
        expected_bytes = (
            seal["student"]["pretrained"]["bytes"]
            if section == "student"
            else seal["teacher"]["checkpoint_bytes"]
        )
        if (
            not path.is_file()
            or sha256_file(path) != expected_sha
            or path.stat().st_size != int(expected_bytes)
        ):
            raise G1PublicationError(
                f"existing sealed {section} private artifact missing/corrupt"
            )
    return seal


def _version_ref(slug: str, version: int) -> str:
    if int(version) < 1:
        raise G1PublicationError(f"invalid Kaggle dataset version: {version}")
    return f"{slug}/{int(version)}"


def _list_version_files(api, version_ref: str) -> list[dict]:
    page_token = None
    rows: list[dict] = []
    for _ in range(100):
        try:
            response = api.dataset_list_files(
                version_ref,
                page_token=page_token,
                page_size=100,
            )
        except Exception as exc:
            raise G1PublicationError(
                f"failed to list exact Kaggle dataset version {version_ref}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        error_message = getattr(response, "error_message", None)
        if error_message:
            raise G1PublicationError(
                f"Kaggle exact-version file listing failed for {version_ref}: "
                f"{error_message}"
            )
        for row in getattr(response, "files", None) or []:
            rows.append(
                {
                    "name": str(getattr(row, "name", "") or ""),
                    "bytes": int(
                        getattr(row, "total_bytes", getattr(row, "size", 0)) or 0
                    ),
                }
            )
        page_token = getattr(response, "next_page_token", None)
        if not page_token:
            return rows
    raise G1PublicationError(
        "Kaggle exact-version file listing exceeded pagination safety bound"
    )


def _require_transport_inventory(
    rows: list[dict], version_ref: str
) -> list[dict]:
    names = [row["name"] for row in rows]
    for required in REQUIRED_REMOTE_FILES:
        count = names.count(required)
        if count != 1:
            raise G1RemoteMismatch(
                f"exact version {version_ref} must contain exactly one {required}; "
                f"observed count={count}, names={names}"
            )
    return rows


def _download_exact_remote_file(
    api, version_ref: str, name: str, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir / ".download-stage" / name.replace("/", "__")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=False)
    try:
        api.dataset_download_file(
            version_ref,
            name,
            path=str(stage),
            force=True,
            quiet=True,
        )
    except Exception as exc:
        raise G1PublicationError(
            f"failed to download {name} from exact version {version_ref}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    files = [path for path in stage.rglob("*") if path.is_file()]
    if len(files) != 1:
        raise G1PublicationError(
            f"Kaggle exact-file download for {name} from {version_ref} "
            f"materialized {len(files)} files; expected exactly one"
        )
    target = output_dir.joinpath(*name.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise G1PublicationError(
            f"refuse to overwrite exact-version download target: {target}"
        )
    shutil.move(str(files[0]), str(target))
    shutil.rmtree(stage)
    return target


def _download_required_transport(
    api, version_ref: str, output_dir: Path
) -> None:
    for name in REQUIRED_REMOTE_FILES:
        _download_exact_remote_file(api, version_ref, name, output_dir)


def _expanded_inventory_names(manifest: dict) -> list[str]:
    return [
        f"{EXPANDED_PACKAGE_DIR}/{row['path']}"
        for row in manifest["members"]
    ]


def _download_expanded_transport(
    api,
    version_ref: str,
    inventory: list[dict],
    output_dir: Path,
) -> G1Package:
    manifest_path = _download_exact_remote_file(
        api, version_ref, MANIFEST_NAME, output_dir
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_package_manifest(manifest)
    if errors:
        raise G1RemoteMismatch(
            f"exact version {version_ref} expanded manifest invalid: "
            + "; ".join(errors)
        )
    expected_names = [MANIFEST_NAME, *_expanded_inventory_names(manifest)]
    observed_names = [row["name"] for row in inventory]
    if sorted(observed_names) != sorted(expected_names):
        raise G1RemoteMismatch(
            f"exact version {version_ref} expanded transport inventory differs "
            f"from manifest; observed={sorted(observed_names)}, "
            f"expected={sorted(expected_names)}"
        )
    for name in expected_names[1:]:
        _download_exact_remote_file(api, version_ref, name, output_dir)
    reconstructed = output_dir.parent / f"{output_dir.name}-reconstructed"
    if reconstructed.exists():
        shutil.rmtree(reconstructed)
    try:
        return reconstruct_expanded_g1_package(output_dir, reconstructed)
    except G1PackageError as exc:
        raise G1RemoteMismatch(
            f"exact version {version_ref} expanded transport reconstruction "
            f"failed: {exc}"
        ) from exc


def _roundtrip_exact_version(
    api,
    slug: str,
    version: int,
    local_package: G1Package,
    seal: dict,
    work_root: Path,
) -> dict:
    version_ref = _version_ref(slug, version)
    inventory = _list_version_files(api, version_ref)
    names = [row["name"] for row in inventory]
    downloaded = work_root / f"downloaded-v{version}"
    if downloaded.exists():
        shutil.rmtree(downloaded)

    if names.count(PACKAGE_NAME) == 1:
        _require_transport_inventory(inventory, version_ref)
        _download_required_transport(api, version_ref, downloaded)
        try:
            remote = locate_g1_package(downloaded)
        except G1PackageError as exc:
            raise G1RemoteMismatch(
                f"exact version {version_ref} transport is invalid: {exc}"
            ) from exc
        remote_transport_mode = "raw_tar"
    elif names.count(PACKAGE_NAME) == 0 and names.count(MANIFEST_NAME) == 1:
        remote = _download_expanded_transport(
            api, version_ref, inventory, downloaded
        )
        remote_transport_mode = "kaggle_expanded_tar"
    else:
        raise G1RemoteMismatch(
            f"exact version {version_ref} has ambiguous G1 transport inventory: "
            f"{names}"
        )

    errors = validate_package_manifest(remote.manifest)
    if errors:
        raise G1RemoteMismatch(
            f"exact version {version_ref} manifest invalid: "
            + "; ".join(errors)
        )
    comparisons = (
        (
            "manifest_sha256",
            remote.manifest.get("manifest_sha256"),
            local_package.manifest.get("manifest_sha256"),
        ),
        (
            "package_sha256",
            remote.manifest.get("package_sha256"),
            local_package.manifest.get("package_sha256"),
        ),
        (
            "package_bytes",
            int(remote.manifest.get("package_bytes", -1)),
            int(local_package.manifest.get("package_bytes", -2)),
        ),
        (
            "g1_seal_sha256",
            remote.manifest.get("g1_seal_sha256"),
            seal.get("g1_seal_sha256"),
        ),
        (
            "source_git_sha",
            remote.manifest.get("source_git_sha"),
            seal.get("source_git_sha"),
        ),
        (
            "dependency_lock_sha256",
            remote.manifest.get("dependency_lock_sha256"),
            seal.get("dependency_lock_sha256"),
        ),
    )
    for field, observed, expected in comparisons:
        if observed != expected:
            raise G1RemoteMismatch(
                f"exact version {version_ref} {field} mismatch: "
                f"observed={observed!r}, expected={expected!r}"
            )

    extracted = work_root / f"roundtrip-extracted-v{version}"
    if extracted.exists():
        shutil.rmtree(extracted)
    safe_extract_g1_package(remote, extracted)

    roundtrip_seal = json.loads(
        (extracted / "G1_MODEL_IDENTITY_SEAL.json").read_text(
            encoding="utf-8"
        )
    )
    if roundtrip_seal != seal:
        raise G1RemoteMismatch(
            "round-trip G1 seal object differs from local sealed G1"
        )

    for key in ("S1", "S2", "S3"):
        row = seal["pair_initializations"][key]
        path = extracted / "private" / row["basename"]
        if (
            sha256_file(path) != row["sha256"]
            or path.stat().st_size != int(row["bytes"])
        ):
            raise G1RemoteMismatch(
                f"round-trip pair verification failed: {key}"
            )

    teacher = extracted / "private" / seal["teacher"]["artifact_basename"]
    if (
        sha256_file(teacher) != seal["teacher"]["checkpoint_sha256"]
        or teacher.stat().st_size != int(seal["teacher"]["checkpoint_bytes"])
    ):
        raise G1RemoteMismatch("round-trip teacher verification failed")

    mnv4 = extracted / "private" / seal["student"]["pretrained"]["basename"]
    if (
        sha256_file(mnv4) != seal["student"]["pretrained"]["sha256"]
        or mnv4.stat().st_size != int(seal["student"]["pretrained"]["bytes"])
    ):
        raise G1RemoteMismatch("round-trip MobileNetV4 verification failed")

    return {
        "published_version_number": int(version),
        "published_version_ref": version_ref,
        "remote_required_file_inventory": inventory,
        "remote_transport_mode": remote_transport_mode,
        "roundtrip_package_sha256": remote.manifest["package_sha256"],
        "roundtrip_member_count": len(remote.manifest["members"]),
        "roundtrip_verified": True,
    }


def wait_for_version_advance(
    api,
    slug: str,
    pre_publish_version_number: int,
    *,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    pre = int(pre_publish_version_number)
    deadline = clock() + timeout_seconds
    last = None
    while clock() < deadline:
        last = _status_snapshot(api, slug)
        status = str(last.get("status", "")).lower()
        current = _version_number(last.get("current_version_number"))
        if status in STATUS_FAILED:
            raise G1PublicationError(
                "Kaggle dataset processing failed while waiting for "
                f"version transition: {last}"
            )
        if current is not None and current < pre:
            raise G1PublicationError(
                f"Kaggle current version regressed from {pre} to {current}"
            )
        if current is not None and current > pre and status in STATUS_READY:
            return last
        sleep_fn(poll_interval_seconds)
    raise G1PublicationError(
        f"Kaggle dataset version did not advance beyond {pre} "
        f"before timeout; last={last}"
    )


def _prepare_transport(
    bundle_dir: str | Path, slug: str, root: Path
) -> G1Package:
    transport = root / "transport"
    transport.mkdir(parents=True, exist_ok=False)
    package = create_g1_package(bundle_dir, transport)
    (transport / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": slug.split("/", 1)[1],
                "id": slug,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    observed = locate_g1_package(transport)
    if observed.manifest != package.manifest:
        raise G1PublicationError(
            "local deterministic G1 transport validation disagrees "
            "with created package"
        )
    return package


def _attempt_base(
    *,
    mode: str,
    slug: str,
    seal: dict,
    package: G1Package,
    preflight: dict,
) -> dict:
    return {
        "schema_version": "1.0",
        "status": "PREPARED",
        "mode": mode,
        "dataset_slug": slug,
        "source_git_sha": seal["source_git_sha"],
        "dependency_lock_sha256": seal["dependency_lock_sha256"],
        "g1_seal_sha256": seal["g1_seal_sha256"],
        "package_sha256": package.manifest["package_sha256"],
        "package_bytes": package.manifest["package_bytes"],
        "package_manifest_sha256": package.manifest["manifest_sha256"],
        "pre_publish_version_number": int(
            preflight["current_version_number"]
        ),
        "mutation_outcome": "NOT_STARTED",
        "scientific_result_produced": False,
        "model_training_performed": False,
    }


def _version_command(transport: Path, seal: dict) -> list[str]:
    return [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(transport),
        "-m",
        f"CropCop sealed G1 {seal['g1_seal_sha256'][:12]}",
        "-q",
        "-r",
        "skip",
    ]


def publish_and_roundtrip(
    bundle_dir: str | Path,
    slug: str,
    receipt_path: str | Path,
    *,
    attempt_path: str | Path | None = None,
    mode: str = "publish",
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[], object] = _api,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    transition_timeout_seconds: float = 900.0,
) -> dict:
    if mode not in {"publish", "repair"}:
        raise G1PublicationError(
            f"unsupported G1 publication mode: {mode}"
        )
    env = os.environ if env is None else env
    seal = validate_local_g1_bundle(bundle_dir)
    preflight = preflight_private_target(
        slug, env=env, api_factory=api_factory
    )
    pre_version = _version_number(
        preflight.get("current_version_number")
    )
    if pre_version is None:
        raise G1PublicationError(
            "authoritative pre-publish version number is unavailable"
        )
    api = api_factory()
    receipt_path = Path(receipt_path).resolve()
    if attempt_path is None:
        attempt_path = receipt_path.with_name(
            "G1_PUBLICATION_ATTEMPT.json"
        )
    attempt_path = Path(attempt_path).resolve()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        package = _prepare_transport(bundle_dir, slug, root)
        attempt = _attempt_base(
            mode=mode,
            slug=slug,
            seal=seal,
            package=package,
            preflight=preflight,
        )
        atomic_write_json(attempt_path, attempt)

        reused_existing = False
        mutation_return_code = None
        mutation_outcome = "NOT_STARTED"
        published_status = None
        roundtrip = None

        if mode == "repair":
            try:
                roundtrip = _roundtrip_exact_version(
                    api, slug, pre_version, package, seal, root
                )
            except G1RemoteMismatch as mismatch:
                attempt["existing_version_match"] = False
                attempt["existing_version_mismatch"] = str(mismatch)
                atomic_write_json(attempt_path, attempt)
            else:
                reused_existing = True
                mutation_outcome = "REUSED_EXISTING_EXACT_VERSION"
                published_status = _status_snapshot(api, slug)

        if roundtrip is None:
            attempt["status"] = "MUTATING"
            attempt["mutation_outcome"] = "IN_PROGRESS"
            atomic_write_json(attempt_path, attempt)
            command = _version_command(root / "transport", seal)
            ambiguous_error = None
            try:
                completed = run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                mutation_return_code = int(
                    getattr(completed, "returncode", 0)
                )
                mutation_outcome = (
                    "COMMAND_RETURNED_ZERO"
                    if mutation_return_code == 0
                    else "COMMAND_NONZERO_AMBIGUOUS"
                )
                if mutation_return_code != 0:
                    ambiguous_error = _redact(
                        (
                            getattr(completed, "stderr", "") or ""
                        ).strip()
                        or (
                            getattr(completed, "stdout", "") or ""
                        ).strip(),
                        env,
                    )
            except subprocess.TimeoutExpired as exc:
                mutation_outcome = "COMMAND_TIMEOUT_AMBIGUOUS"
                ambiguous_error = _redact(str(exc), env)
            except Exception as exc:
                mutation_outcome = "COMMAND_EXCEPTION_AMBIGUOUS"
                ambiguous_error = _redact(
                    f"{type(exc).__name__}: {exc}", env
                )

            attempt.update(
                {
                    "status": "MUTATION_RETURNED",
                    "mutation_outcome": mutation_outcome,
                    "mutation_return_code": mutation_return_code,
                    "mutation_diagnostic": (
                        ambiguous_error or ""
                    )[-1600:],
                }
            )
            atomic_write_json(attempt_path, attempt)

            try:
                published_status = wait_for_version_advance(
                    api,
                    slug,
                    pre_version,
                    timeout_seconds=transition_timeout_seconds,
                )
            except Exception as exc:
                try:
                    observed = _status_snapshot(api, slug)
                except Exception as status_exc:
                    observed = {
                        "status_probe_error": (
                            f"{type(status_exc).__name__}: "
                            f"{status_exc}"
                        )
                    }
                safe_next = (
                    "run explicit g1-publication-repair with the exact "
                    "existing sealed G1 bundle; do not regenerate pairs "
                    "or seal"
                )
                attempt.update(
                    {
                        "status": "FAILED",
                        "stage": "PUBLICATION_VERSION_ADVANCE",
                        "observed_remote_state": observed,
                        "safe_next_action": safe_next,
                    }
                )
                atomic_write_json(attempt_path, attempt)
                raise G1PublicationError(
                    "G1 publication version transition failed: "
                    + json.dumps(
                        {
                            "stage": "PUBLICATION_VERSION_ADVANCE",
                            "slug": slug,
                            "pre_version": pre_version,
                            "observed_current_version": observed.get(
                                "current_version_number"
                            ),
                            "local_package_sha": package.manifest[
                                "package_sha256"
                            ],
                            "remote_state": observed,
                            "mutation_outcome": mutation_outcome,
                            "safe_next_action": safe_next,
                        },
                        sort_keys=True,
                    )
                ) from exc

            published_version = _version_number(
                published_status.get("current_version_number")
            )
            if (
                published_version is None
                or published_version <= pre_version
            ):
                raise G1PublicationError(
                    "version-transition barrier returned without "
                    "an advanced version"
                )
            attempt.update(
                {
                    "status": "VERSION_ADVANCED",
                    "published_version_number": published_version,
                    "published_version_ref": _version_ref(
                        slug, published_version
                    ),
                    "processing_status": published_status.get(
                        "status"
                    ),
                }
            )
            atomic_write_json(attempt_path, attempt)
            roundtrip = _roundtrip_exact_version(
                api, slug, published_version, package, seal, root
            )

        attempt.update(
            {
                "status": "ROUNDTRIP_PASS",
                "mutation_outcome": mutation_outcome,
                "published_version_number": roundtrip[
                    "published_version_number"
                ],
                "published_version_ref": roundtrip[
                    "published_version_ref"
                ],
                "roundtrip_verified": True,
            }
        )
        atomic_write_json(attempt_path, attempt)

        receipt = {
            "schema_version": "2.0",
            "status": "PASS",
            "dataset_slug": slug,
            "pre_publish_version_number": pre_version,
            "published_version_number": roundtrip[
                "published_version_number"
            ],
            "published_version_ref": roundtrip[
                "published_version_ref"
            ],
            "processing_status": (
                published_status or {}
            ).get("status"),
            "g1_seal_sha256": seal["g1_seal_sha256"],
            "package_sha256": package.manifest["package_sha256"],
            "package_bytes": package.manifest["package_bytes"],
            "package_manifest_sha256": package.manifest[
                "manifest_sha256"
            ],
            "roundtrip_package_sha256": roundtrip[
                "roundtrip_package_sha256"
            ],
            "roundtrip_member_count": roundtrip[
                "roundtrip_member_count"
            ],
            "roundtrip_verified": True,
            "authenticated_owner_match": preflight["owner_match"],
            "authoritative_is_private": preflight[
                "authoritative_is_private"
            ],
            "remote_required_file_inventory": roundtrip[
                "remote_required_file_inventory"
            ],
            "remote_transport_mode": roundtrip.get(
                "remote_transport_mode", "raw_tar"
            ),
            "mutation_outcome": mutation_outcome,
            "mutation_return_code": mutation_return_code,
            "reused_existing_exact_version": reused_existing,
        }
        atomic_write_json(receipt_path, receipt)
        return receipt
