from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .trackb_eaai_common import TrackBEAAIError, sha256_file

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def discover_images(
    root: Path,
    allowed_labels: set[str],
) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for path in sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ):
        label = path.parent.name
        if label not in allowed_labels:
            raise TrackBEAAIError(
                f"unexpected source label {label!r}: {path}"
            )
        rows.append((path, label))
    return rows


def read_historical_sha_set(
    path: Path,
) -> tuple[set[str], int]:
    hashes: set[str] = set()
    row_count = 0
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "raw_sha256" not in (reader.fieldnames or []):
            raise TrackBEAAIError(
                "historical manifest lacks raw_sha256"
            )
        for row in reader:
            value = str(row["raw_sha256"]).strip().lower()
            if (
                len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise TrackBEAAIError(
                    "historical manifest contains malformed SHA-256"
                )
            hashes.add(value)
            row_count += 1
    if row_count != 92744:
        raise TrackBEAAIError(
            "historical manifest row count mismatch: "
            f"expected=92744 got={row_count}"
        )
    return hashes, row_count


def build_external_manifest(
    *,
    dataset_id: str,
    data_root: Path,
    mapping: dict[str, str],
    expected_count: int,
    expected_support: dict[str, int] | None,
    historical_sha: set[str],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image, ImageOps

    items = discover_images(data_root, set(mapping))
    if len(items) != int(expected_count):
        raise TrackBEAAIError(
            f"{dataset_id}: expected {expected_count} images, "
            f"found {len(items)}"
        )

    support = Counter(label for _, label in items)
    if expected_support is not None:
        expected = {
            str(key): int(value)
            for key, value in expected_support.items()
        }
        if dict(support) != expected:
            raise TrackBEAAIError(
                f"{dataset_id}: source support mismatch: "
                f"observed={dict(support)} expected={expected}"
            )

    def one(item: tuple[Path, str]) -> dict[str, Any]:
        path, label = item
        raw_sha = sha256_file(path)
        with Image.open(path) as image:
            canonical = ImageOps.exif_transpose(image).convert("RGB")
            canonical.load()
            width, height = canonical.size

        rel = path.relative_to(data_root).as_posix()
        row_id = hashlib.sha256(
            f"{dataset_id}|{rel}|{raw_sha}".encode("utf-8")
        ).hexdigest()
        return {
            "row_id": row_id,
            "relative_path": rel,
            "source_label": label,
            "target_class_name": mapping[label],
            "raw_sha256": raw_sha,
            "bytes": path.stat().st_size,
            "width": int(width),
            "height": int(height),
            "exact_v1_train_val_overlap": raw_sha in historical_sha,
        }

    with ThreadPoolExecutor(
        max_workers=max(1, int(workers))
    ) as pool:
        rows = list(pool.map(one, items, chunksize=16))

    rows.sort(key=lambda row: row["row_id"])

    by_sha: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_sha[row["raw_sha256"]].append(row["row_id"])

    representatives = {
        raw_sha: min(row_ids)
        for raw_sha, row_ids in by_sha.items()
    }
    for row in rows:
        ids = by_sha[row["raw_sha256"]]
        row["exact_duplicate_group_size"] = len(ids)
        row["exact_duplicate_representative"] = (
            row["row_id"]
            == representatives[row["raw_sha256"]]
        )

    overlap_rows = [
        row for row in rows
        if row["exact_v1_train_val_overlap"]
    ]
    clean_rows = [
        row for row in rows
        if not row["exact_v1_train_val_overlap"]
    ]
    deduplicated_rows = [
        row for row in clean_rows
        if row["exact_duplicate_representative"]
    ]
    duplicate_groups = [
        ids for ids in by_sha.values()
        if len(ids) > 1
    ]

    audit = {
        "dataset_id": dataset_id,
        "published_row_count": len(rows),
        "source_support": dict(sorted(support.items())),
        "exact_v1_train_val_overlap_count": len(overlap_rows),
        "leakage_clean_primary_count": len(clean_rows),
        "exact_deduplicated_sensitivity_count": len(
            deduplicated_rows
        ),
        "external_exact_duplicate_group_count": len(
            duplicate_groups
        ),
        "external_rows_in_duplicate_groups": sum(
            len(group) for group in duplicate_groups
        ),
        "v1_test_accessed": False,
    }
    return rows, audit


def write_manifest_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_id",
        "relative_path",
        "source_label",
        "target_class_name",
        "raw_sha256",
        "bytes",
        "width",
        "height",
        "exact_v1_train_val_overlap",
        "exact_duplicate_group_size",
        "exact_duplicate_representative",
    ]
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
