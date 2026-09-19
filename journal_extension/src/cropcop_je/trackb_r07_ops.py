from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hashing import sha256_file
from .publication import audit_public_files, publish_to_github_branch


class TrackBOpsError(RuntimeError):
    pass


KAGGLE_OWNER_DEFAULT = "ranamuhammadahmed6"
KAGGLE_HISTORICAL_DATASET = f"{KAGGLE_OWNER_DEFAULT}/cropcop-trackb-historical-r07-v2"
KAGGLE_EVIDENCE_DATASET = f"{KAGGLE_OWNER_DEFAULT}/cropcop-trackb-r07-evidence"

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
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception as exc:
        raise TrackBOpsError("Kaggle UserSecretsClient is unavailable") from exc
    try:
        value = UserSecretsClient().get_secret(name)
    except Exception as exc:
        raise TrackBOpsError(f"required Kaggle secret is unavailable: {name}") from exc
    value = (value or "").strip()
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


def publish_private_kaggle_dataset(
    *,
    folder: str | Path,
    slug: str,
    title: str,
    version_message: str,
    license_name: str = "other",
) -> dict[str, str | bool]:
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
    existed = kaggle_dataset_exists(slug)
    if existed:
        run_checked(
            ["kaggle", "datasets", "version", "-p", str(folder), "-m", version_message, "-q", "-r", "zip"],
            timeout=7200,
        )
    else:
        # Kaggle CLI dataset creation is private unless --public/-u is explicitly supplied.
        run_checked(
            ["kaggle", "datasets", "create", "-p", str(folder), "-q", "-r", "zip"],
            timeout=7200,
        )
    return {"slug": slug, "created": not existed, "versioned": existed}


def _urlopen_json(url: str, *, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "CropCop-TrackB/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = response.read()
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise TrackBOpsError(f"JSON object expected from {url}")
    return obj


def _stream_download(url: str, destination: Path, *, timeout: int = 1800) -> dict[str, str | int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "CropCop-TrackB/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response, destination.open("wb") as fh:
        content_disposition = response.headers.get("Content-Disposition", "")
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            fh.write(block)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "content_disposition": content_disposition,
    }


def _safe_extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise TrackBOpsError(f"unsafe archive member: {info.filename}")
        zf.extractall(destination)


def acquire_irish_potato(destination: str | Path) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
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
    for name in ("earlyblt.zip", "healthy.zip", "lateblt.zip"):
        row = by_name[name]
        links = row.get("links") or {}
        url = links.get("self") or links.get("content")
        if not url:
            raise TrackBOpsError(f"Zenodo file has no download URL: {name}")
        archive = destination / "_transport" / name
        receipt = _stream_download(str(url), archive)
        class_name = name[:-4]
        _safe_extract_zip(archive, data_root / class_name)
        transport.append({"name": name, **receipt})
        archive.unlink(missing_ok=True)
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
        "known_historical_contributor_relationship": False,
        "lineage_review_status": "RESIDUAL_UNCERTAINTY",
        "acquisition_transport": transport,
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


def _normalize_gvlid_tree(extracted_root: Path, data_root: Path) -> dict[str, int]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    supports: dict[str, int] = {"Black Rot": 0, "Esca": 0, "Healthy": 0, "Leaf Blight": 0}
    used: set[Path] = set()
    for path in sorted(extracted_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
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
        target = data_root / label / f"{supports[label]:06d}_{path.name}"
        _hardlink_or_copy(path, target)
        supports[label] += 1
    if sum(supports.values()) != 3477:
        raise TrackBOpsError(f"GVLiD v5 expected 3477 images, observed {supports} total={sum(supports.values())}")
    if any(value < 50 for value in supports.values()):
        raise TrackBOpsError(f"GVLiD class support below frozen minimum: {supports}")
    return supports


def acquire_gvlid_v5(destination: str | Path) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
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
    for idx, url in enumerate(links):
        parsed = urllib.parse.urlparse(url)
        name = Path(parsed.path).name or f"mendeley_{idx:03d}.bin"
        if name == "file_downloaded":
            name = f"mendeley_{idx:03d}.zip"
        target = transport_dir / name
        receipt = _stream_download(url, target)
        receipts.append({"url": url, **receipt})
        try:
            if zipfile.is_zipfile(target):
                _safe_extract_zip(target, extracted / f"part_{idx:03d}")
            else:
                raise TrackBOpsError(f"GVLiD public-file transport is not a ZIP archive: {target}")
        finally:
            target.unlink(missing_ok=True)
    data_root = destination / "data"
    supports = _normalize_gvlid_tree(extracted, data_root)
    shutil.rmtree(extracted, ignore_errors=True)
    source = {
        "doi": "10.17632/wkymf8bhcg.5",
        "version": "5",
        "source_url": page_url,
        "retrieved_at": utc_now(),
        "license_or_access_text": "CC BY 4.0",
        "known_historical_contributor_relationship": False,
        "lineage_review_status": "RESIDUAL_UNCERTAINTY",
        "source_page_sha256": page_sha,
        "observed_class_support": supports,
        "published_total_images": 3477,
        "published_class_count_table_status": "NOT_USED_AS_AUTHORITY_DUE_ONE_IMAGE_ARITHMETIC_DISCREPANCY",
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
    if len(closure_sha) != 64:
        raise TrackBOpsError("terminal closure lacks valid closure_sha256")
    run_id = f"TB2-FINAL-{closure_sha[:16]}"
    with tempfile.TemporaryDirectory() as td:
        staged = create_publication_staging(output_root, Path(td) / "public")
        branch = publish_to_github_branch(
            repo_dir=repo_root,
            source_git_sha=source_git_sha,
            run_id=run_id,
            files=[str(p) for p in staged],
            destination_prefix="journal_extension/evidence/public/track_b",
        )
    return {"run_id": run_id, "branch": branch, "closure_sha256": closure_sha}


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
