from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath

from .data import ManifestRow
from .hashing import require_sha256
from .surfaces import TRAIN, VAL, authorize_training_surface

FROZEN_V1_COLUMNS = {
    "row_id": "record_key",
    "relative_path": "portable_relpath",
    "split": "split",
    "label": "label",
}


def validate_frozen_v1_column_contract(
    *,
    row_id_column: str,
    path_column: str,
    split_column: str,
    label_column: str,
    class_index_column: str = "",
) -> None:
    observed = {
        "row_id": str(row_id_column),
        "relative_path": str(path_column),
        "split": str(split_column),
        "label": str(label_column),
    }
    if observed != FROZEN_V1_COLUMNS:
        raise ValueError(
            "frozen Final-V1 manifest column contract mismatch: "
            f"expected={FROZEN_V1_COLUMNS}, observed={observed}"
        )
    if str(class_index_column).strip():
        raise ValueError(
            "frozen Final-V1 has no numeric class-index column; "
            "targets must be derived from the hash-locked class_to_idx.json"
        )


def _load_locked_class_map(path: str | Path, expected_sha256: str) -> dict[str, int]:
    require_sha256(path, expected_sha256, "120-way class map")
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or len(obj) != 120:
        raise ValueError("frozen class_to_idx.json must be a direct 120-entry mapping")
    out: dict[str, int] = {}
    for label, value in obj.items():
        if not isinstance(label, str) or not label:
            raise ValueError("class map contains an invalid label key")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"class map index for {label!r} is not an integer")
        if not 0 <= value < 120:
            raise ValueError(f"class map index outside frozen 120-way range: {label!r} -> {value}")
        out[label] = value
    if set(out.values()) != set(range(120)):
        raise ValueError("class map indices are not an exact bijection over 0..119")
    return out


def load_frozen_v1_rows(
    manifest_path: str | Path,
    class_map_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_class_map_sha256: str,
    surface: str,
    row_id_column: str = "record_key",
    path_column: str = "portable_relpath",
    split_column: str = "split",
    label_column: str = "label",
    class_index_column: str = "",
    train_split_value: str = "train",
    val_split_value: str = "val",
    expected_count: int | None = None,
) -> list[ManifestRow]:
    validate_frozen_v1_column_contract(
        row_id_column=row_id_column,
        path_column=path_column,
        split_column=split_column,
        label_column=label_column,
        class_index_column=class_index_column,
    )
    surface = authorize_training_surface(surface)
    require_sha256(manifest_path, expected_manifest_sha256, "V1 manifest")
    class_to_idx = _load_locked_class_map(class_map_path, expected_class_map_sha256)
    split_value = train_split_value if surface == TRAIN else val_split_value

    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    with Path(manifest_path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = set(FROZEN_V1_COLUMNS.values())
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"frozen Final-V1 manifest is missing required columns: {sorted(missing)}")
        for raw in reader:
            if raw[split_column] != split_value:
                continue
            row_id = str(raw[row_id_column]).strip()
            relative_path = str(raw[path_column]).strip()
            label = str(raw[label_column]).strip()
            if not row_id or row_id in seen_ids:
                raise ValueError(f"missing/duplicate stable row ID on {surface}: {row_id!r}")
            if not relative_path or relative_path in seen_paths:
                raise ValueError(f"missing/duplicate portable path on {surface}: {relative_path!r}")
            posix = PurePosixPath(relative_path)
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"unsafe portable path on {surface}: {relative_path!r}")
            if not relative_path.startswith(split_value + "/"):
                raise ValueError(
                    f"portable path split mismatch on {surface}: "
                    f"split={split_value!r}, path={relative_path!r}"
                )
            if label not in class_to_idx:
                raise ValueError(f"manifest label absent from frozen class map: {label!r}")
            seen_ids.add(row_id)
            seen_paths.add(relative_path)
            rows.append(
                ManifestRow(
                    stable_row_id=row_id,
                    relative_path=relative_path,
                    split=raw[split_column],
                    class_index=class_to_idx[label],
                )
            )

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(
            f"{surface} row count mismatch: expected {expected_count}, got {len(rows)}"
        )
    return rows
