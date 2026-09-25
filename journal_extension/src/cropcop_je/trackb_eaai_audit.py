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
    from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

    # Fail closed on truncated/corrupt payloads. We report and exclude them;
    # we never ask Pillow to synthesize pixels from a damaged source file.
    ImageFile.LOAD_TRUNCATED_IMAGES = False

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

    def one(
        item: tuple[Path, str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        path, label = item
        raw_sha = sha256_file(path)
        rel = path.relative_to(data_root).as_posix()
        row_id = hashlib.sha256(
            f"{dataset_id}|{rel}|{raw_sha}".encode("utf-8")
        ).hexdigest()
        common = {
            "row_id": row_id,
            "relative_path": rel,
            "source_label": label,
            "target_class_name": mapping[label],
            "raw_sha256": raw_sha,
            "bytes": path.stat().st_size,
            "exact_v1_train_val_overlap": raw_sha in historical_sha,
        }

        try:
            with Image.open(path) as image:
                canonical = ImageOps.exif_transpose(image).convert("RGB")
                canonical.load()
                width, height = canonical.size
        except (
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            invalid = {
                **common,
                "decode_status": "INVALID",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
            return None, invalid

        row = {
            **common,
            "width": int(width),
            "height": int(height),
            "decode_status": "VALID",
        }
        return row, None

    with ThreadPoolExecutor(
        max_workers=max(1, int(workers))
    ) as pool:
        scanned = list(pool.map(one, items, chunksize=16))

    rows = [row for row, invalid in scanned if row is not None]
    invalid_rows = [
        invalid for row, invalid in scanned if invalid is not None
    ]
    rows.sort(key=lambda row: row["row_id"])
    invalid_rows.sort(key=lambda row: row["row_id"])

    valid_support = Counter(row["source_label"] for row in rows)
    missing_valid_labels = sorted(set(mapping) - set(valid_support))
    if missing_valid_labels:
        raise TrackBEAAIError(
            f"{dataset_id}: mapped labels have no strictly decodable images: "
            f"{missing_valid_labels}"
        )

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
        "published_file_count": len(items),
        "published_source_support": dict(sorted(support.items())),
        "decode_valid_row_count": len(rows),
        "decode_valid_support": dict(sorted(valid_support.items())),
        "decode_invalid_count": len(invalid_rows),
        "decode_invalid_exact_v1_train_val_overlap_count": sum(
            int(row["exact_v1_train_val_overlap"]) for row in invalid_rows
        ),
        "decode_invalid_support": dict(
            sorted(Counter(row["source_label"] for row in invalid_rows).items())
        ),
        "decode_policy": {
            "strict_decode_required": True,
            "load_truncated_images": False,
            "decode_invalid_files_excluded_from_metrics": True,
        },
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
    return rows, invalid_rows, audit


def write_invalid_manifest_csv(
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
        "decode_status",
        "error_type",
        "exact_v1_train_val_overlap",
        "error_message",
    ]
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
        "decode_status",
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
