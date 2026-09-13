from __future__ import annotations

import csv
import os
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    PRINCIPAL_G1_SEAL_SHA256,
    OperatorError,
    load_json,
    sha256_file,
)


MANIFEST_NAMES = ("final_manifest.csv", "dataset_manifest.csv")
CLASS_MAP_NAMES = (
    "class_index.json",
    "class_map.json",
    "class-map.json",
    "classes.json",
    "class_names.json",
    "class_to_idx.json",
)
METADATA_DIR_NAMES = {
    "audit", "metadata", "manifests", "reports", "report", "certification",
    "certification_reports", "cropcop_final_v1_certification_reports",
}
SHALLOW_MAX_DEPTH = 5
MAX_DIAGNOSTIC_FILES = 40


def _mounted_top_level(input_root: Path) -> list[str]:
    try:
        return sorted(p.name for p in input_root.iterdir())
    except OSError:
        return []


def _shallow_dirs(input_root: Path, max_depth: int = SHALLOW_MAX_DEPTH) -> list[Path]:
    """Enumerate only shallow dataset/package directories; never descend into image files."""
    root = input_root.resolve()
    rows = [root]
    frontier = [(root, 0)]
    seen = {root}
    while frontier:
        parent, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        try:
            children = [p.resolve() for p in parent.iterdir() if p.is_dir()]
        except (OSError, PermissionError):
            children = []
        for child in sorted(children, key=str):
            if child in seen:
                continue
            seen.add(child)
            rows.append(child)
            frontier.append((child, depth + 1))
    return rows


def _candidate_named_files(input_root: Path, names: tuple[str, ...]) -> list[Path]:
    rows: list[Path] = []
    for directory in _shallow_dirs(input_root):
        for name in names:
            path = directory / name
            try:
                if path.is_file():
                    rows.append(path.resolve())
            except OSError:
                continue
    return sorted(set(rows), key=str)


def _metadata_files(input_root: Path) -> list[Path]:
    """Collect shallow metadata files without walking class/image directories."""
    root = input_root.resolve()
    rows: list[Path] = []
    for directory in _shallow_dirs(root):
        try:
            depth = len(directory.relative_to(root).parts)
        except ValueError:
            continue
        inspect_files = depth <= 4 or directory.name.casefold() in METADATA_DIR_NAMES
        if not inspect_files:
            continue
        try:
            entries = list(directory.iterdir())
        except (OSError, PermissionError):
            continue
        for path in entries:
            try:
                if path.is_file() and path.suffix.casefold() in {".json", ".jsonl", ".csv"}:
                    rows.append(path.resolve())
            except OSError:
                continue
    return sorted(set(rows), key=str)


def _manifest_sample_paths(manifest: Path, limit: int = 16) -> list[str]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "portable_relpath" not in (reader.fieldnames or []):
            raise OperatorError("frozen manifest is missing portable_relpath")
        rows: list[str] = []
        for row in reader:
            rel = str(row.get("portable_relpath", "")).strip()
            if rel:
                rows.append(rel)
            if len(rows) >= limit:
                break
    if len(rows) < min(8, limit):
        raise OperatorError("manifest did not provide enough image paths for root qualification")
    return rows


def _nearest_image_root(manifest: Path, input_root: Path, override: str = "") -> tuple[Path, int]:
    sample = _manifest_sample_paths(manifest)
    if override:
        candidate = Path(override).resolve()
        if not all((candidate / rel).is_file() for rel in sample):
            raise OperatorError("manual CROPCOP_IMAGE_ROOT does not resolve the frozen manifest sample")
        return candidate, 0

    base = input_root.resolve()
    current = manifest.parent.resolve()
    distance = 0
    while True:
        if all((current / rel).is_file() for rel in sample):
            return current, distance
        if current == base or base not in current.parents:
            break
        current = current.parent
        distance += 1
    raise OperatorError(f"manifest copy does not co-resolve with its image tree: {manifest}")


def _exact_hash_candidates(paths: list[Path], expected_sha: str) -> list[Path]:
    matches: list[Path] = []
    for path in paths:
        try:
            if sha256_file(path) == expected_sha:
                matches.append(path.resolve())
        except (OSError, PermissionError):
            continue
    return sorted(set(matches), key=str)


def _resolve_manifest_and_image_root(input_root: Path) -> tuple[Path, Path]:
    override = str(os.environ.get("CROPCOP_MANIFEST", "") or "").strip()
    image_override = str(os.environ.get("CROPCOP_IMAGE_ROOT", "") or "").strip()
    if override:
        manifests = [Path(override).resolve()]
        if not manifests[0].is_file() or sha256_file(manifests[0]) != MANIFEST_SHA256:
            raise OperatorError("manual CROPCOP_MANIFEST has wrong SHA-256 or is not a file")
    else:
        named = _candidate_named_files(input_root, MANIFEST_NAMES)
        manifests = _exact_hash_candidates(named, MANIFEST_SHA256)
        if not manifests:
            raise OperatorError(
                f"no shallow manifest candidate matched sha256={MANIFEST_SHA256}; "
                f"named candidates={[str(p) for p in named]}"
            )

    qualified: list[tuple[int, int, str, Path, Path]] = []
    rejected: list[str] = []
    for manifest in manifests:
        try:
            image_root, distance = _nearest_image_root(manifest, input_root, override=image_override)
        except OperatorError as exc:
            rejected.append(f"{manifest}: {exc}")
            continue
        inside_root = 0 if image_root == manifest.parent or image_root in manifest.parents else 1
        qualified.append((distance, inside_root, str(manifest), manifest, image_root))

    if not qualified:
        raise OperatorError(
            "exact frozen manifest bytes were found, but no copy co-resolved with the mounted image tree; "
            f"exact_copies={[str(p) for p in manifests]}; rejected={rejected}"
        )
    qualified.sort(key=lambda row: (row[0], row[1], row[2]))
    best = qualified[0]
    if len(qualified) > 1:
        first_score = best[:2]
        tied = [row for row in qualified if row[:2] == first_score]
        if len(tied) > 1 and len({row[4] for row in tied}) > 1:
            raise OperatorError(
                "multiple byte-identical manifest copies independently qualify different image roots; "
                f"set CROPCOP_MANIFEST/CROPCOP_IMAGE_ROOT explicitly: "
                f"{[(str(row[3]), str(row[4])) for row in tied]}"
            )
        print(
            "Frozen manifest duplicate copies detected; selecting structurally nearest dataset-tree copy:",
            best[3],
        )
    return best[3], best[4]


def _resolve_class_map(input_root: Path, manifest: Path, image_root: Path) -> Path:
    override = str(os.environ.get("CROPCOP_CLASS_MAP", "") or "").strip()
    if override:
        path = Path(override).resolve()
        if not path.is_file() or sha256_file(path) != CLASS_MAP_SHA256:
            raise OperatorError("manual CROPCOP_CLASS_MAP has wrong SHA-256 or is not a file")
        return path

    named = _candidate_named_files(input_root, CLASS_MAP_NAMES)
    matches = _exact_hash_candidates(named, CLASS_MAP_SHA256)
    if not matches:
        # Fallback is still bounded to shallow metadata locations and small files only.
        fallback: list[Path] = []
        for path in _metadata_files(input_root):
            try:
                if path.stat().st_size <= 16 * 1024 * 1024:
                    fallback.append(path)
            except OSError:
                continue
        matches = _exact_hash_candidates(fallback, CLASS_MAP_SHA256)
    if not matches:
        raise OperatorError(f"no shallow class-map candidate matched sha256={CLASS_MAP_SHA256}")

    def score(path: Path) -> tuple[int, int, str]:
        under_image_root = 0 if image_root == path.parent or image_root in path.parents else 1
        try:
            common = len(os.path.commonpath([str(path), str(manifest)]))
        except ValueError:
            common = 0
        return (under_image_root, -common, str(path))

    matches.sort(key=score)
    if len(matches) > 1:
        print("Frozen class-map duplicate copies detected; selecting dataset-nearest copy:", matches[0])
    return matches[0]


def _resolve_principal_g1_fast(input_root: Path) -> Path:
    override = str(os.environ.get("CROPCOP_PRINCIPAL_G1", "") or "").strip()
    roots = [Path(override).resolve()] if override else [
        path.parent.resolve()
        for path in _candidate_named_files(input_root, ("G1_MODEL_IDENTITY_SEAL.json",))
    ]
    matches: list[Path] = []
    for root in roots:
        seal_path = root / "G1_MODEL_IDENTITY_SEAL.json"
        if not seal_path.is_file():
            continue
        try:
            seal = load_json(seal_path)
        except Exception:
            continue
        if seal.get("g1_seal_sha256") != PRINCIPAL_G1_SEAL_SHA256:
            continue
        if not (root / "private").is_dir() or not (root / "evidence").is_dir():
            continue
        matches.append(root)
    matches = sorted(set(matches), key=str)
    if len(matches) != 1:
        raise OperatorError(
            "principal historical G1 bundle auto-resolution must be unique; "
            f"found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]


def _candidate_metadata(input_root: Path, limit: int = MAX_DIAGNOSTIC_FILES) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _metadata_files(input_root):
        if len(rows) >= limit:
            break
        name = path.name.casefold()
        likely = (
            name in {x.casefold() for x in MANIFEST_NAMES + CLASS_MAP_NAMES}
            or "manifest" in name
            or "class" in name
            or name.endswith("_seal.json")
        )
        if not likely:
            continue
        row: dict[str, object] = {"path": str(path), "bytes": None, "sha256": None}
        try:
            size = path.stat().st_size
            row["bytes"] = size
            if size <= 64 * 1024 * 1024:
                row["sha256"] = sha256_file(path)
            else:
                row["sha256"] = "<skipped-large-diagnostic>"
        except (OSError, PermissionError):
            row["sha256"] = "<unreadable>"
        rows.append(row)
    return rows


def _dataset_error_message(exc: Exception, input_root: Path) -> str:
    mounted = _mounted_top_level(input_root)
    candidates = _candidate_metadata(input_root)
    return (
        "Frozen CropCop V1 input preflight failed before dependency installation.\n"
        f"Original error: {exc}\n"
        f"Required dataset_manifest SHA256: {MANIFEST_SHA256}\n"
        f"Required class_index SHA256: {CLASS_MAP_SHA256}\n"
        f"Mounted /kaggle/input entries: {mounted or ['<none>']}\n"
        f"Candidate metadata files (path/bytes/sha256): {candidates or ['<none>']}\n"
        "Action: attach the frozen CropCop V1 dataset. Byte-identical metadata duplicates are allowed only "
        "when one copy structurally co-resolves with the image tree. Manual CROPCOP_MANIFEST, "
        "CROPCOP_CLASS_MAP and CROPCOP_IMAGE_ROOT overrides remain exact-hash/structure verified."
    )


def resolve_master_inputs(account_id: str, input_root: str | Path = "/kaggle/input") -> tuple[Path, Path, Path]:
    root = Path(input_root)
    try:
        manifest, image_root = _resolve_manifest_and_image_root(root)
        class_map = _resolve_class_map(root, manifest, image_root)
    except (OperatorError, OSError) as exc:
        raise OperatorError(_dataset_error_message(exc, root)) from exc

    print("Frozen V1 input preflight PASS")
    print("  manifest:", manifest)
    print("  class map:", class_map)
    print("  image root:", image_root)

    if account_id == "K1":
        try:
            principal = _resolve_principal_g1_fast(root)
        except (OperatorError, OSError) as exc:
            mounted = _mounted_top_level(root)
            candidates = _candidate_metadata(root)
            raise OperatorError(
                "K1 historical principal-G1 preflight failed before dependency installation.\n"
                f"Original error: {exc}\n"
                f"Mounted /kaggle/input entries: {mounted or ['<none>']}\n"
                f"Candidate metadata files: {candidates or ['<none>']}\n"
                "Action: attach the complete historical principal G1 bundle (seal + private/ + evidence/), "
                "or set CROPCOP_PRINCIPAL_G1 to its exact mounted directory."
            ) from exc
        print("Historical principal G1 preflight PASS:", principal)

    return manifest, class_map, image_root
