from __future__ import annotations

import os
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    OperatorError,
    resolve_frozen_dataset,
    resolve_image_root,
    resolve_principal_g1_bundle,
    sha256_file,
)


LIKELY_METADATA_NAMES = {
    "dataset_manifest.csv",
    "class_index.json",
    "final_cropcop_registry.csv",
    "g1_model_identity_seal.json",
}


def _mounted_top_level(input_root: Path) -> list[str]:
    try:
        return sorted(p.name for p in input_root.iterdir())
    except OSError:
        return []


def _candidate_metadata(input_root: Path, limit: int = 40) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        iterator = input_root.rglob("*")
    except OSError:
        return rows
    for path in iterator:
        if len(rows) >= limit:
            break
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        name = path.name.casefold()
        likely = (
            name in LIKELY_METADATA_NAMES
            or "manifest" in name
            or "class_index" in name
            or "class-map" in name
            or "class_map" in name
            or name.endswith("_seal.json")
        )
        if not likely:
            continue
        row: dict[str, object] = {"path": str(path), "bytes": None, "sha256": None}
        try:
            row["bytes"] = path.stat().st_size
            row["sha256"] = sha256_file(path)
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
        "Action: attach the frozen CropCop V1 Kaggle dataset containing the exact manifest, class map, "
        "and image tree. If duplicate mounts are intentional, set CROPCOP_MANIFEST, CROPCOP_CLASS_MAP, "
        "and CROPCOP_IMAGE_ROOT to the exact mounted paths; overrides are still hash/structure verified."
    )


def resolve_master_inputs(account_id: str, input_root: str | Path = "/kaggle/input") -> tuple[Path, Path, Path]:
    root = Path(input_root)
    try:
        manifest, class_map = resolve_frozen_dataset(
            input_root=root,
            manifest_override=os.environ.get("CROPCOP_MANIFEST", ""),
            class_map_override=os.environ.get("CROPCOP_CLASS_MAP", ""),
        )
        image_root = resolve_image_root(
            manifest,
            input_root=root,
            override=os.environ.get("CROPCOP_IMAGE_ROOT", ""),
        )
    except (OperatorError, OSError) as exc:
        raise OperatorError(_dataset_error_message(exc, root)) from exc

    print("Frozen V1 input preflight PASS")
    print("  manifest:", manifest)
    print("  class map:", class_map)
    print("  image root:", image_root)

    if account_id == "K1":
        try:
            principal = resolve_principal_g1_bundle(
                root=root,
                override=os.environ.get("CROPCOP_PRINCIPAL_G1", ""),
            )
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
