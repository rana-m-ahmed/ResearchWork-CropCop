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
    G1PackageError,
    create_g1_package,
    locate_g1_package,
    safe_extract_g1_package,
)
from .hashing import sha256_file

STATUS_READY = {"ready", "complete", "completed"}


class G1PublicationError(RuntimeError):
    pass


def _required(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name, "")).strip()
    if not value:
        raise G1PublicationError(f"required Kaggle publication credential missing: {name}")
    return value


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


def preflight_private_target(
    slug: str,
    *,
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[], object] = _api,
) -> dict:
    env = os.environ if env is None else env
    slug = validate_private_slug(slug)
    username = _required(env, "KAGGLE_USERNAME")
    _required(env, "KAGGLE_KEY")
    owner, dataset_slug = slug.split("/", 1)
    if owner.casefold() != username.casefold():
        raise G1PublicationError(
            f"G1 private target owner mismatch: slug owner {owner!r} != authenticated KAGGLE_USERNAME {username!r}"
        )

    api = api_factory()
    try:
        metadata = _metadata(api, slug)
    except Exception as exc:
        if str(env.get("CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET", "")).strip() != "1":
            raise G1PublicationError(
                "G1 private target does not exist or metadata is inaccessible. "
                "Pre-create a private Kaggle Dataset with this exact slug, or explicitly set "
                "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        raise G1PublicationError(
            "explicit private-target creation was requested but must be performed by "
            "create_private_target_if_missing() before publication"
        ) from exc

    if metadata.get("isPrivate") is not True:
        raise G1PublicationError("G1 target is not authoritatively reported private by Kaggle metadata")
    meta_id = str(metadata.get("id") or metadata.get("ref") or "")
    if meta_id and meta_id.casefold() != slug.casefold():
        raise G1PublicationError(f"Kaggle metadata identity mismatch: {meta_id!r} != {slug!r}")
    refs = _mine_refs(api, dataset_slug)
    if slug not in refs and slug.casefold() not in {ref.casefold() for ref in refs}:
        raise G1PublicationError(
            "G1 target is not present in the authenticated account's Kaggle 'mine' dataset set"
        )
    status = json.loads(api.dataset_status(slug, format="json"))
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "dataset_slug": slug,
        "authenticated_username": username,
        "owner_match": True,
        "mine_membership": True,
        "authoritative_is_private": True,
        "dataset_status": status.get("status"),
        "current_version_number": status.get("current_version_number"),
    }


def create_private_target_if_missing(
    slug: str,
    *,
    env: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    env = os.environ if env is None else env
    slug = validate_private_slug(slug)
    if str(env.get("CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET", "")).strip() != "1":
        raise G1PublicationError("private target auto-creation requires CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1")
    username = _required(env, "KAGGLE_USERNAME")
    _required(env, "KAGGLE_KEY")
    owner, dataset_slug = slug.split("/", 1)
    if owner.casefold() != username.casefold():
        raise G1PublicationError("refuse to create G1 private target under a different owner")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "README.txt").write_text(
            "Private CropCop G1 transport target. Real sealed package versions are uploaded separately.\n",
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
            ) + "\n",
            encoding="utf-8",
        )
        run(
            ["kaggle", "datasets", "create", "-p", str(root), "-q", "-r", "skip"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )


def wait_until_ready(api, slug: str, *, timeout_seconds: float = 900.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        payload = json.loads(api.dataset_status(slug, format="json"))
        last = payload
        status = str(payload.get("status", "")).lower()
        if status in STATUS_READY:
            return payload
        if status in {"error", "failed", "failure"}:
            raise G1PublicationError(f"Kaggle dataset processing failed: {payload}")
        time.sleep(5)
    raise G1PublicationError(f"Kaggle dataset did not become ready before timeout; last={last}")


def validate_local_g1_bundle(bundle_dir: str | Path) -> dict:
    bundle = Path(bundle_dir).resolve()
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file():
        raise G1PublicationError("G1_MODEL_IDENTITY_SEAL.json is missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    errors = validate_g1_seal_object(seal)
    if errors:
        raise G1PublicationError("existing G1 bundle seal is invalid: " + "; ".join(errors))
    for key in ("S1", "S2", "S3"):
        row = seal["pair_initializations"][key]
        path = bundle / "private" / row["basename"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise G1PublicationError(f"existing G1 pair artifact missing/corrupt: {key}")
    for section, basename in (
        ("student", seal["student"]["pretrained"]["basename"]),
        ("teacher", seal["teacher"]["artifact_basename"]),
    ):
        path = bundle / "private" / basename
        expected = (
            seal["student"]["pretrained"]["sha256"]
            if section == "student"
            else seal["teacher"]["checkpoint_sha256"]
        )
        if not path.is_file() or sha256_file(path) != expected:
            raise G1PublicationError(f"existing sealed {section} private artifact missing/corrupt")
    return seal


def publish_and_roundtrip(
    bundle_dir: str | Path,
    slug: str,
    receipt_path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    api_factory: Callable[[], object] = _api,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    env = os.environ if env is None else env
    seal = validate_local_g1_bundle(bundle_dir)
    preflight = preflight_private_target(slug, env=env, api_factory=api_factory)
    api = api_factory()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        transport = root / "transport"
        transport.mkdir()
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
            ) + "\n",
            encoding="utf-8",
        )
        run(
            [
                "kaggle", "datasets", "version",
                "-p", str(transport),
                "-m", f"CropCop sealed G1 {seal['g1_seal_sha256'][:12]}",
                "-q", "-r", "skip",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        status = wait_until_ready(api, slug)
        downloaded = root / "downloaded"
        downloaded.mkdir()
        api.dataset_download_files(slug, path=str(downloaded), force=True, quiet=True, unzip=True)
        roundtrip = locate_g1_package(downloaded)
        if roundtrip.manifest["package_sha256"] != package.manifest["package_sha256"]:
            raise G1PublicationError("published G1 package SHA differs after Kaggle round-trip")
        extracted = root / "roundtrip-extracted"
        safe_extract_g1_package(roundtrip, extracted)

        roundtrip_seal = json.loads(
            (extracted / "G1_MODEL_IDENTITY_SEAL.json").read_text(encoding="utf-8")
        )
        if roundtrip_seal != seal:
            raise G1PublicationError("round-trip G1 seal bytes/object differ")
        for key in ("S1", "S2", "S3"):
            row = seal["pair_initializations"][key]
            path = extracted / "private" / row["basename"]
            if sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["bytes"]):
                raise G1PublicationError(f"round-trip pair verification failed: {key}")
        teacher = extracted / "private" / seal["teacher"]["artifact_basename"]
        if sha256_file(teacher) != seal["teacher"]["checkpoint_sha256"]:
            raise G1PublicationError("round-trip teacher verification failed")
        mnv4 = extracted / "private" / seal["student"]["pretrained"]["basename"]
        if sha256_file(mnv4) != seal["student"]["pretrained"]["sha256"]:
            raise G1PublicationError("round-trip MobileNetV4 verification failed")

        receipt = {
            "schema_version": "1.0",
            "status": "PASS",
            "dataset_slug": slug,
            "authoritative_is_private": preflight["authoritative_is_private"],
            "authenticated_owner_match": preflight["owner_match"],
            "edit_access_evidence": "authenticated owner + mine membership + successful version upload",
            "processing_status": status.get("status"),
            "current_version_number": status.get("current_version_number"),
            "g1_seal_sha256": seal["g1_seal_sha256"],
            "package_sha256": package.manifest["package_sha256"],
            "package_manifest_sha256": package.manifest["manifest_sha256"],
            "roundtrip_package_sha256": roundtrip.manifest["package_sha256"],
            "roundtrip_member_count": len(roundtrip.manifest["members"]),
            "roundtrip_verified": True,
        }
        atomic_write_json(receipt_path, receipt)
        return receipt
