from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .atomic_io import atomic_write_json
from .hashing import sha256_file, sha256_json

PACKAGE_NAME = "G1_PACKAGE.tar"
MANIFEST_NAME = "G1_PACKAGE_MANIFEST.json"
READINESS_PACKAGE_NAME = "G1_READINESS_TRANSPORT.tar"


class G1PackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class G1Package:
    package_path: Path
    manifest_path: Path
    manifest: dict


def _safe_rel(name: str) -> PurePosixPath:
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or ".." in rel.parts or "." in rel.parts:
        raise G1PackageError(f"unsafe G1 package member path: {name!r}")
    if str(rel) != name:
        raise G1PackageError(f"non-canonical G1 package member path: {name!r}")
    return rel


def _tree_files(root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink():
            raise G1PackageError(f"symlink forbidden in G1 package transport: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise G1PackageError(f"unsupported G1 package member type: {path}")
        rel = path.relative_to(root).as_posix()
        _safe_rel(rel)
        files.append((rel, path))
    if not files:
        raise G1PackageError("G1 package transport source tree is empty")
    return files


def _bundle_files(bundle_dir: Path) -> list[tuple[str, Path]]:
    files = _tree_files(bundle_dir)
    if "G1_MODEL_IDENTITY_SEAL.json" not in {name for name, _ in files}:
        raise G1PackageError("G1 package source lacks G1_MODEL_IDENTITY_SEAL.json")
    return files


def _write_deterministic_tar(
    files: list[tuple[str, Path]],
    package_path: Path,
) -> list[dict]:
    rows = [
        {
            "path": rel,
            "type": "file",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for rel, path in files
    ]
    with tarfile.open(package_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for rel, path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(data))
    return rows


def _manifest_without_hash(manifest: dict) -> dict:
    clean = dict(manifest)
    clean.pop("manifest_sha256", None)
    return clean


def validate_package_manifest(manifest: dict) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "1.0":
        errors.append("unsupported G1 package manifest schema")
    if manifest.get("package_basename") != PACKAGE_NAME:
        errors.append("G1 package basename mismatch")
    if len(str(manifest.get("package_sha256", ""))) != 64:
        errors.append("G1 package SHA invalid")
    if int(manifest.get("package_bytes", 0) or 0) <= 0:
        errors.append("G1 package byte count invalid")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        errors.append("G1 package manifest has no members")
    else:
        seen = set()
        for row in members:
            name = str(row.get("path", ""))
            try:
                _safe_rel(name)
            except Exception as exc:
                errors.append(str(exc))
                continue
            if name in seen:
                errors.append(f"duplicate G1 package member: {name}")
            seen.add(name)
            if len(str(row.get("sha256", ""))) != 64:
                errors.append(f"G1 package member SHA invalid: {name}")
            if int(row.get("bytes", -1)) < 0:
                errors.append(f"G1 package member byte count invalid: {name}")
            if row.get("type") != "file":
                errors.append(f"G1 package manifest permits non-file member: {name}")
    expected = sha256_json(_manifest_without_hash(manifest))
    if manifest.get("manifest_sha256") != expected:
        errors.append("G1 package manifest self-hash mismatch")
    return errors


def create_g1_package(
    bundle_dir: str | Path,
    output_dir: str | Path,
) -> G1Package:
    bundle = Path(bundle_dir).resolve()
    output = Path(output_dir).resolve()
    if not bundle.is_dir():
        raise G1PackageError(f"G1 bundle directory missing: {bundle}")
    output.mkdir(parents=True, exist_ok=True)
    package_path = output / PACKAGE_NAME
    manifest_path = output / MANIFEST_NAME
    if package_path.exists() or manifest_path.exists():
        raise G1PackageError("refuse to overwrite existing G1 package transport")

    files = _bundle_files(bundle)
    rows = _write_deterministic_tar(files, package_path)

    seal = json.loads((bundle / "G1_MODEL_IDENTITY_SEAL.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "1.0",
        "transport": "deterministic_uncompressed_tar",
        "package_basename": PACKAGE_NAME,
        "package_sha256": sha256_file(package_path),
        "package_bytes": package_path.stat().st_size,
        "g1_seal_sha256": seal.get("g1_seal_sha256"),
        "source_git_sha": seal.get("source_git_sha"),
        "dependency_lock_sha256": seal.get("dependency_lock_sha256"),
        "members": rows,
    }
    manifest["manifest_sha256"] = sha256_json(_manifest_without_hash(manifest))
    atomic_write_json(manifest_path, manifest)
    return G1Package(package_path=package_path, manifest_path=manifest_path, manifest=manifest)


def readiness_transport_dry_run(
    staging_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    staging = Path(staging_dir).resolve()
    output = Path(output_dir).resolve()
    if not staging.is_dir():
        raise G1PackageError(f"readiness staging directory missing: {staging}")
    output.mkdir(parents=True, exist_ok=True)
    package_path = output / READINESS_PACKAGE_NAME
    extracted = output / "readiness-extracted"
    if package_path.exists() or (extracted.exists() and any(extracted.iterdir())):
        raise G1PackageError("readiness transport target must be fresh")

    files = _tree_files(staging)
    rows = _write_deterministic_tar(files, package_path)
    expected = {row["path"]: row for row in rows}
    extracted.mkdir(parents=True, exist_ok=True)
    observed: set[str] = set()

    with tarfile.open(package_path, "r:") as archive:
        for member in archive.getmembers():
            name = member.name
            _safe_rel(name)
            if name in observed:
                raise G1PackageError(f"duplicate readiness tar member: {name}")
            observed.add(name)
            if name not in expected:
                raise G1PackageError(f"unexpected readiness tar member: {name}")
            if not member.isfile() or member.issym() or member.islnk():
                raise G1PackageError(f"non-regular readiness tar member forbidden: {name}")
            row = expected[name]
            if member.size != int(row["bytes"]):
                raise G1PackageError(f"readiness tar member byte count mismatch: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise G1PackageError(f"cannot read readiness tar member: {name}")
            target = extracted.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            if sha256_file(target) != row["sha256"]:
                raise G1PackageError(f"readiness tar member SHA mismatch: {name}")

    if observed != set(expected):
        raise G1PackageError(
            f"readiness tar member set mismatch; missing={sorted(set(expected) - observed)}"
        )
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "transport": "deterministic_uncompressed_tar",
        "package_basename": READINESS_PACKAGE_NAME,
        "package_sha256": sha256_file(package_path),
        "package_bytes": package_path.stat().st_size,
        "member_count": len(rows),
        "members": rows,
        "safe_extract_verified": True,
        "production_g1_seal_created": False,
        "pair_initializations_created": False,
    }


def locate_g1_package(input_root: str | Path) -> G1Package:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise G1PackageError(f"CROPCOP_G1_INPUT_ROOT is not a directory: {root}")
    package = root / PACKAGE_NAME
    manifest_path = root / MANIFEST_NAME
    if not package.is_file() or not manifest_path.is_file():
        raise G1PackageError(
            f"G1 input root must contain exact {PACKAGE_NAME} + {MANIFEST_NAME} at its root"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_package_manifest(manifest)
    if errors:
        raise G1PackageError("G1 package manifest invalid: " + "; ".join(errors))
    if sha256_file(package) != manifest["package_sha256"]:
        raise G1PackageError("G1 package SHA differs from manifest")
    if package.stat().st_size != int(manifest["package_bytes"]):
        raise G1PackageError("G1 package byte count differs from manifest")
    return G1Package(package_path=package, manifest_path=manifest_path, manifest=manifest)


def safe_extract_g1_package(
    package: G1Package,
    destination: str | Path,
) -> Path:
    dest = Path(destination).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise G1PackageError(f"G1 extraction target must be empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    expected = {row["path"]: row for row in package.manifest["members"]}
    observed: set[str] = set()
    with tarfile.open(package.package_path, "r:") as archive:
        for member in archive.getmembers():
            name = member.name
            _safe_rel(name)
            if name in observed:
                raise G1PackageError(f"duplicate member inside G1 tar: {name}")
            observed.add(name)
            if name not in expected:
                raise G1PackageError(f"unexpected member inside G1 tar: {name}")
            if not member.isfile() or member.issym() or member.islnk():
                raise G1PackageError(f"non-regular member forbidden in G1 tar: {name}")
            row = expected[name]
            if member.size != int(row["bytes"]):
                raise G1PackageError(f"G1 tar member byte count mismatch: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise G1PackageError(f"cannot read G1 tar member: {name}")
            data = source.read()
            target = dest.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if sha256_file(target) != row["sha256"]:
                raise G1PackageError(f"G1 tar member SHA mismatch after extraction: {name}")

    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise G1PackageError(f"G1 tar members missing: {missing}")

    for name, row in expected.items():
        path = dest.joinpath(*PurePosixPath(name).parts)
        if not path.is_file():
            raise G1PackageError(f"extracted G1 package member missing: {name}")
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise G1PackageError(f"post-extract G1 member verification failed: {name}")
    return dest


def mount_g1_input(
    input_root: str | Path,
    working_root: str | Path,
) -> tuple[Path, G1Package]:
    package = locate_g1_package(input_root)
    bundle = Path(working_root).resolve()
    safe_extract_g1_package(package, bundle)
    return bundle, package
