from __future__ import annotations

import csv
import os
from pathlib import Path, PurePosixPath

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
SPLIT_DIR_NAMES = {"train", "val", "test"}
SKIP_IMAGE_ROOT_DESCENT = METADATA_DIR_NAMES | SPLIT_DIR_NAMES | {
    "private", "evidence", "__pycache__", ".git",
}
SHALLOW_MAX_DEPTH = 7
IMAGE_ROOT_MAX_DEPTH = 9
MAX_DIAGNOSTIC_FILES = 40
MANIFEST_SAMPLE_PER_SPLIT = 6


def _mounted_top_level(input_root: Path) -> list[str]:
    try:
        return sorted(p.name for p in input_root.iterdir())
    except OSError:
        return []


def _shallow_dirs(input_root: Path, max_depth: int = SHALLOW_MAX_DEPTH) -> list[Path]:
    """Enumerate shallow package directories without intentionally walking image/class trees."""
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
            if child.name.casefold() in SPLIT_DIR_NAMES:
                continue
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
    """Collect bounded metadata candidates; never recursively enumerate image files."""
    root = input_root.resolve()
    rows: list[Path] = []
    for directory in _shallow_dirs(root):
        try:
            depth = len(directory.relative_to(root).parts)
        except ValueError:
            continue
        inspect_files = depth <= 5 or directory.name.casefold() in METADATA_DIR_NAMES
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


def _manifest_sample_paths(manifest: Path, per_split: int = MANIFEST_SAMPLE_PER_SPLIT) -> list[str]:
    """Sample only TRAIN/VAL paths; the protected test image surface is never opened here."""
    wanted: dict[str, list[str]] = {"train": [], "val": []}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"portable_relpath", "split"}
        missing = required.difference(fields)
        if missing:
            raise OperatorError(f"frozen manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            split = str(row.get("split", "")).strip()
            if split not in wanted:
                continue
            rel = str(row.get("portable_relpath", "")).strip()
            if not rel:
                continue
            posix = PurePosixPath(rel)
            if posix.is_absolute() or ".." in posix.parts or not rel.startswith(split + "/"):
                raise OperatorError(f"unsafe or split-mismatched portable_relpath: {rel!r}")
            if len(wanted[split]) < per_split:
                wanted[split].append(rel)
            if all(len(rows) >= per_split for rows in wanted.values()):
                break
    counts = {split: len(rows) for split, rows in wanted.items()}
    if any(count < per_split for count in counts.values()):
        raise OperatorError(f"manifest did not provide enough TRAIN/VAL paths for root qualification: {counts}")
    return wanted["train"] + wanted["val"]


def _candidate_image_roots(input_root: Path, max_depth: int = IMAGE_ROOT_MAX_DEPTH) -> list[Path]:
    """Find shallow directories whose direct children include train/ and val/."""
    root = input_root.resolve()
    rows: list[Path] = []
    frontier = [(root, 0)]
    seen = {root}
    while frontier:
        parent, depth = frontier.pop(0)
        try:
            has_train = (parent / "train").is_dir()
            has_val = (parent / "val").is_dir()
        except OSError:
            has_train = has_val = False
        if has_train and has_val:
            rows.append(parent)
            continue
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
            if child.name.casefold() in SKIP_IMAGE_ROOT_DESCENT:
                continue
            frontier.append((child, depth + 1))
    return sorted(set(rows), key=str)


def _root_affinity(manifest: Path, image_root: Path) -> tuple[int, int, int, str]:
    """Lower tuple is better: more shared ancestry, avoid report trees, then shorter distance."""
    try:
        common = Path(os.path.commonpath([str(manifest.parent.resolve()), str(image_root.resolve())]))
        common_depth = len(common.parts)
    except (ValueError, OSError):
        common_depth = 0
    report_penalty = sum(
        1 for part in image_root.parts
        if part.casefold() in METADATA_DIR_NAMES or "certification_report" in part.casefold()
    )
    distance = len(manifest.parent.resolve().parts) + len(image_root.resolve().parts) - 2 * common_depth
    return (-common_depth, report_penalty, distance, str(image_root.resolve()))


def _qualify_image_root(
    manifest: Path,
    candidates: list[Path],
    override: str = "",
) -> tuple[Path, tuple[int, int, int, str]]:
    sample = _manifest_sample_paths(manifest)
    if override:
        candidate = Path(override).resolve()
        if not candidate.is_dir():
            raise OperatorError("manual CROPCOP_IMAGE_ROOT is not a directory")
        missing = [rel for rel in sample if not (candidate / rel).is_file()]
        if missing:
            raise OperatorError(
                "manual CROPCOP_IMAGE_ROOT does not resolve the frozen TRAIN/VAL manifest sample; "
                f"first_missing={missing[:4]}"
            )
        return candidate, _root_affinity(manifest, candidate)

    qualified = [candidate for candidate in candidates if all((candidate / rel).is_file() for rel in sample)]
    if not qualified:
        raise OperatorError(
            f"manifest copy does not resolve against any discovered TRAIN/VAL image root: {manifest}; "
            f"discovered_roots={[str(p) for p in candidates[:16]]}"
        )
    ranked = sorted((_root_affinity(manifest, root), root) for root in qualified)
    best_score, best_root = ranked[0]
    ties = [root for score, root in ranked if score[:3] == best_score[:3]]
    if len(set(ties)) > 1:
        raise OperatorError(
            "one manifest copy qualifies multiple equally-ranked image roots; "
            f"set CROPCOP_IMAGE_ROOT explicitly: {[str(p) for p in ties]}"
        )
    return best_root, best_score


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
    manifest_override = str(os.environ.get("CROPCOP_MANIFEST", "") or "").strip()
    image_override = str(os.environ.get("CROPCOP_IMAGE_ROOT", "") or "").strip()

    if manifest_override:
        manifests = [Path(manifest_override).resolve()]
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

    root_candidates = [] if image_override else _candidate_image_roots(input_root)
    if not image_override and not root_candidates:
        raise OperatorError(
            "exact frozen manifest bytes were found, but no shallow directory with direct train/ and val/ "
            "children was discovered under /kaggle/input"
        )

    qualified: list[tuple[tuple[int, int, int, str], str, Path, Path]] = []
    rejected: list[str] = []
    for manifest in manifests:
        try:
            image_root, score = _qualify_image_root(manifest, root_candidates, override=image_override)
        except OperatorError as exc:
            rejected.append(f"{manifest}: {exc}")
            continue
        qualified.append((score, str(manifest), manifest, image_root))

    if not qualified:
        raise OperatorError(
            "exact frozen manifest bytes were found, but no copy resolved with the mounted TRAIN/VAL image tree; "
            f"exact_copies={[str(p) for p in manifests]}; "
            f"discovered_image_roots={[str(p) for p in root_candidates[:16]]}; rejected={rejected}"
        )

    qualified.sort(key=lambda row: (row[0], row[1]))
    best = qualified[0]
    best_score = best[0][:3]
    tied = [row for row in qualified if row[0][:3] == best_score]
    if len(tied) > 1 and len({row[3] for row in tied}) > 1:
        raise OperatorError(
            "multiple byte-identical manifest copies independently qualify equally-ranked different image roots; "
            f"set CROPCOP_MANIFEST/CROPCOP_IMAGE_ROOT explicitly: "
            f"{[(str(row[2]), str(row[3])) for row in tied]}"
        )

    if len(manifests) > 1:
        print("Frozen manifest duplicate copies detected; selected dataset-nearest copy:", best[2])
    print("Resolved frozen TRAIN/VAL image root:", best[3])
    return best[2], best[3]


def _common_depth(a: Path, b: Path) -> int:
    try:
        return len(Path(os.path.commonpath([str(a.resolve()), str(b.resolve())])).parts)
    except (ValueError, OSError):
        return 0


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

    def score(path: Path) -> tuple[int, int, int, str]:
        manifest_common = _common_depth(path.parent, manifest.parent)
        image_common = _common_depth(path.parent, image_root)
        report_penalty = sum(
            1 for part in path.parts
            if part.casefold() in METADATA_DIR_NAMES or "certification_report" in part.casefold()
        )
        return (-manifest_common, -image_common, report_penalty, str(path))

    matches.sort(key=score)
    best = matches[0]
    best_score = score(best)[:3]
    tied = [path for path in matches if score(path)[:3] == best_score]
    if len(tied) > 1 and len(set(tied)) > 1:
        family_depths = {_common_depth(path.parent, manifest.parent) for path in tied}
        if len(family_depths) != 1:
            raise OperatorError(
                "multiple exact class-map copies are structurally ambiguous; "
                f"set CROPCOP_CLASS_MAP explicitly: {[str(p) for p in tied]}"
            )
    if len(matches) > 1:
        print("Frozen class-map duplicate copies detected; selecting dataset-nearest copy:", best)
    return best


def _resolve_principal_g1_fast(input_root: Path) -> Path:
    override = str(os.environ.get("CROPCOP_PRINCIPAL_G1", "") or "").strip()
    roots = [Path(override).resolve()] if override else [
        path.parent.resolve()
        for path in _candidate_named_files(input_root, ("G1_MODEL_IDENTITY_SEAL.json",))
    ]
    matches: list[Path] = []
    rejected: list[str] = []
    for root in roots:
        seal_path = root / "G1_MODEL_IDENTITY_SEAL.json"
        if not seal_path.is_file():
            continue
        try:
            seal = load_json(seal_path)
        except Exception as exc:
            rejected.append(f"{seal_path}: unreadable JSON ({type(exc).__name__})")
            continue
        if seal.get("g1_seal_sha256") != PRINCIPAL_G1_SEAL_SHA256:
            rejected.append(f"{seal_path}: g1_seal_sha256 mismatch")
            continue
        if not (root / "private").is_dir() or not (root / "evidence").is_dir():
            rejected.append(f"{root}: missing private/ or evidence/")
            continue
        matches.append(root)
    matches = sorted(set(matches), key=str)
    if len(matches) != 1:
        raise OperatorError(
            "principal historical G1 bundle auto-resolution must be unique and complete; "
            f"found {len(matches)}: {[str(x) for x in matches]}; rejected={rejected[:12]}"
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
    split_roots = _candidate_image_roots(input_root)
    return (
        "Frozen CropCop V1 input preflight failed before dependency installation.\n"
        f"Original error: {exc}\n"
        f"Required dataset_manifest SHA256: {MANIFEST_SHA256}\n"
        f"Required class_index SHA256: {CLASS_MAP_SHA256}\n"
        f"Mounted /kaggle/input entries: {mounted or ['<none>']}\n"
        f"Discovered TRAIN/VAL image-root candidates: {[str(p) for p in split_roots[:16]] or ['<none>']}\n"
        f"Candidate metadata files (path/bytes/sha256): {candidates or ['<none>']}\n"
        "Action: keep the frozen V1 dataset attached. The resolver accepts byte-identical metadata copies, "
        "discovers nested TRAIN/VAL split roots, and selects the copy with strongest structural affinity. "
        "Manual CROPCOP_MANIFEST, CROPCOP_CLASS_MAP and CROPCOP_IMAGE_ROOT overrides remain exact-hash/"
        "structure verified."
    )


def resolve_master_inputs(account_id: str, input_root: str | Path = "/kaggle/input") -> tuple[Path, Path, Path]:
    root = Path(input_root)
    try:
        manifest, image_root = _resolve_manifest_and_image_root(root)
        class_map = _resolve_class_map(root, manifest, image_root)
    except (OperatorError, OSError) as exc:
        raise OperatorError(_dataset_error_message(exc, root)) from exc

    os.environ["CROPCOP_MANIFEST"] = str(manifest)
    os.environ["CROPCOP_CLASS_MAP"] = str(class_map)
    os.environ["CROPCOP_IMAGE_ROOT"] = str(image_root)

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
        os.environ["CROPCOP_PRINCIPAL_G1"] = str(principal)
        print("Historical principal G1 preflight PASS:", principal)

    return manifest, class_map, image_root
