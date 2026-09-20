from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import ctc_v2_eval_transform
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.models import load_exact_teacher
from cropcop_je.tracka_v12_evidence import validate_selected_checkpoint_replay
from cropcop_je.tracka_v12_posttraining import clean_replay, prediction_rows, LockedValidationDataset, write_jsonl
from cropcop_je.trackb_r07 import (
    AUTHORITY_ID,
    CLASS_MAP_SHA256,
    DATASET_MANIFEST_SHA256,
    DINO_AUDIT_SHA256,
    DINO_FACTORY_MANIFEST_SHA256,
    R07_CHECKPOINTS,
    R07_RUN_RECORDS,
    TrackBError,
    audit_policy_from_lock,
    assign_candidate_grade,
    build_candidate_seal,
    discover_kaggle_inputs,
    load_json,
    load_r07_checkpoint,
    mapped_scope_metrics,
    resolve_bundle_file,
    three_seed_summary,
    validate_downstream_authority,
    validate_execution_lock,
    validate_prior_attempt_for_rerun,
    validate_same_prediction_surface,
    verify_candidate_seal,
    verify_code_attestation,
)
from cropcop_je.trackb_r07_analysis import bootstrap_three_seed_macro_f1
from cropcop_je.trackb_r07_ops import publish_attempt_state, read_latest_attempt_state, utc_now
from cropcop_je.trackb_r07_audit import (
    ImageAuditRecord,
    build_family_components,
    cross_hash_pairs,
    discover_candidate_images,
    deterministic_representative_order,
    exact_duplicate_pairs,
    make_image_record,
    near_hash_pairs,
    orb_features_from_path,
    representative_manifest,
    topk_cosine_neighbors,
    verify_orb_pair,
    write_jsonl as audit_write_jsonl,
    encode_audit_features,
)

VAL_COUNT = 16368


def stage(name: str):
    print(f"\n{'=' * 78}\nTRACK B :: {name}\n{'=' * 78}", flush=True)


def _json_file(bundle, key):
    return load_json(resolve_bundle_file(bundle, key))


def _resolve_dir(bundle, field: str) -> Path:
    rel = str(bundle.manifest.get(field, "")).strip()
    if not rel:
        raise TrackBError(f"input role {bundle.role} is missing directory field {field}")
    path = (bundle.root / rel).resolve()
    if bundle.root not in path.parents and path != bundle.root:
        raise TrackBError(f"input role {bundle.role} directory {field} escapes dataset root")
    if not path.is_dir():
        raise TrackBError(f"input role {bundle.role} directory missing: {path}")
    return path


def _class_map(path: Path) -> dict[str, int]:
    from cropcop_je.hashing import require_sha256
    require_sha256(path, CLASS_MAP_SHA256, "frozen 120-way class map")
    obj = load_json(path)
    if len(obj) != 120 or set(int(v) for v in obj.values()) != set(range(120)):
        raise TrackBError("class map is not the frozen 120-way bijection")
    return {str(k): int(v) for k, v in obj.items()}


def _parallel_records(items, root: Path, workers: int) -> tuple[list[ImageAuditRecord], list[dict[str, Any]]]:
    import hashlib
    def one(item):
        path, label = item
        path = Path(path).resolve()
        try:
            return ("ok", make_image_record(path, root, source_label=label))
        except Exception as exc:
            rel = path.relative_to(root).as_posix()
            raw_sha = sha256_file(path) if path.is_file() else ""
            row_id = hashlib.sha256(f"{rel}|{raw_sha}".encode("utf-8")).hexdigest() if raw_sha else ""
            return ("fail", {
                "row_id": row_id,
                "relative_path": rel,
                "source_label": str(label),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": raw_sha,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        output = list(pool.map(one, items, chunksize=8))
    records = [value for status, value in output if status == "ok"]
    failures = [value for status, value in output if status == "fail"]
    return records, failures


def _write_candidate_raw_manifest(path: Path, records: list[ImageAuditRecord], failures: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "relative_path", "source_label", "bytes", "sha256", "decode_status", "decode_error", "phash64", "dhash64", "width", "height"]
    rows = []
    for row in records:
        rows.append({
            "row_id": row.row_id, "relative_path": row.relative_path, "source_label": row.source_label,
            "bytes": row.bytes, "sha256": row.sha256, "decode_status": "PASS", "decode_error": "",
            "phash64": row.phash64, "dhash64": row.dhash64, "width": row.width, "height": row.height,
        })
    for row in failures:
        rows.append({
            "row_id": row.get("row_id", ""), "relative_path": row["relative_path"], "source_label": row["source_label"],
            "bytes": row.get("bytes", 0), "sha256": row.get("sha256", ""), "decode_status": "FAIL",
            "decode_error": f"{row.get('error_type','')}: {row.get('error','')}",
            "phash64": "", "dhash64": "", "width": "", "height": "",
        })
    rows.sort(key=lambda row: row["relative_path"])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _read_hist_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"hist_id", "raw_sha256", "phash64", "dhash64", "width", "height"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise TrackBError(f"historical comparison manifest missing columns: {sorted(missing)}")
        for raw in reader:
            rows.append({
                "hist_id": str(raw["hist_id"]),
                "raw_sha256": str(raw["raw_sha256"]),
                "phash64": int(raw["phash64"]),
                "dhash64": int(raw["dhash64"]),
                "width": int(raw["width"]),
                "height": int(raw["height"]),
            })
    if not rows or len({r["hist_id"] for r in rows}) != len(rows):
        raise TrackBError(f"historical comparison surface must contain unique non-empty rows, got {len(rows)}")
    return rows


def _load_hist_orb(bundle, hist_rows):
    import numpy as np
    offsets = np.load(resolve_bundle_file(bundle, "orb_offsets"), mmap_mode="r")
    xy = np.load(resolve_bundle_file(bundle, "orb_xy"), mmap_mode="r")
    desc = np.load(resolve_bundle_file(bundle, "orb_desc"), mmap_mode="r")
    shapes = np.load(resolve_bundle_file(bundle, "orb_shapes"), mmap_mode="r")
    if offsets.shape != (len(hist_rows) + 1,) or shapes.shape != (len(hist_rows), 2):
        raise TrackBError("historical ORB index shape mismatch")
    if int(offsets[-1]) != len(xy) or len(xy) != len(desc) or desc.shape[1:] != (32,):
        raise TrackBError("historical ORB packed arrays are inconsistent")
    def get(index: int):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        return {
            "xy": np.asarray(xy[lo:hi], dtype=np.float32),
            "desc": np.asarray(desc[lo:hi], dtype=np.uint8),
            "shape": tuple(int(x) for x in shapes[index]),
        }
    return get


def _load_dino(core):
    checkpoint = resolve_bundle_file(core, "dino_checkpoint")
    if sha256_file(checkpoint) != DINO_AUDIT_SHA256:
        raise TrackBError("DINO audit encoder bytes do not match the frozen historical reference")
    factory_manifest = resolve_bundle_file(core, "dino_factory_manifest")
    factory_record = load_json(factory_manifest)
    factory_spec = str(factory_record.get("entrypoint", ""))
    if not factory_spec or ":" not in factory_spec:
        raise TrackBError("DINO factory bundle does not define a valid entrypoint")
    source_root_rel = str(core.manifest.get("dino_factory_source_root", "")).strip()
    source_root = (core.root / source_root_rel).resolve() if source_root_rel else core.root
    model, identity = load_exact_teacher(
        checkpoint,
        factory_spec=factory_spec,
        factory_bundle_manifest=factory_manifest,
        repo_root=source_root,
        factory_source_root=source_root,
    )
    return model, identity


def _self_dino_pairs(features, row_ids: list[str], *, device: str, top_k: int) -> set[tuple[str, str]]:
    import numpy as np
    if len(row_ids) < 2:
        return set()
    k = min(int(top_k) + 1, len(row_ids))
    idx, _ = topk_cosine_neighbors(features, features, k=k, device=device)
    pairs = set()
    for i in range(len(row_ids)):
        kept = 0
        for j in idx[i]:
            j = int(j)
            if j == i:
                continue
            a, b = sorted((row_ids[i], row_ids[j]))
            if a != b:
                pairs.add((a, b))
                kept += 1
            if kept >= int(top_k):
                break
    return pairs


def _cross_exact_pairs(external_records, hist_rows):
    by_sha: dict[str, list[str]] = {}
    for row in hist_rows:
        by_sha.setdefault(row["raw_sha256"], []).append(row["hist_id"])
    return {(ext.row_id, hist_id) for ext in external_records for hist_id in by_sha.get(ext.sha256, [])}


def _candidate_orb_getter(
    records: list[ImageAuditRecord],
    root: Path,
    workers: int,
    audit_policy,
    cache_root: Path,
):
    """Build a bounded file-backed ORB cache instead of retaining all descriptors in RAM."""
    import numpy as np

    cache_root = Path(cache_root).resolve()
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=False)
    chunk_root = cache_root / ".chunks"
    chunk_root.mkdir()

    ordered = list(records)
    row_index = {row.row_id: index for index, row in enumerate(ordered)}
    offsets = [0]
    chunk_paths: list[Path] = []
    chunk_size = 256

    for chunk_index, start_index in enumerate(range(0, len(ordered), chunk_size)):
        subset = ordered[start_index:start_index + chunk_size]

        def build(row):
            return orb_features_from_path(
                root / row.relative_path,
                max_side=audit_policy.orb_max_side,
                nfeatures=audit_policy.orb_nfeatures,
            )

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            computed = list(pool.map(build, subset, chunksize=8))

        shapes = np.asarray([orb["shape"] for orb in computed], dtype=np.int32)
        counts = np.asarray([len(orb["desc"]) for orb in computed], dtype=np.int32)
        xy = (
            np.concatenate([orb["xy"] for orb in computed], axis=0)
            if int(counts.sum())
            else np.empty((0, 2), dtype=np.float32)
        )
        desc = (
            np.concatenate([orb["desc"] for orb in computed], axis=0)
            if int(counts.sum())
            else np.empty((0, 32), dtype=np.uint8)
        )
        chunk_path = chunk_root / f"chunk_{chunk_index:05d}.npz"
        np.savez(
            chunk_path,
            shapes=shapes,
            counts=counts,
            xy=xy.astype(np.float32, copy=False),
            desc=desc.astype(np.uint8, copy=False),
        )
        chunk_paths.append(chunk_path)
        for count in counts.tolist():
            offsets.append(offsets[-1] + int(count))
        print(
            f"Candidate ORB cache {min(start_index + len(subset), len(ordered))}/{len(ordered)}",
            flush=True,
        )

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    np.save(cache_root / "offsets.npy", offsets_arr)
    total = int(offsets_arr[-1])
    xy_mm = np.lib.format.open_memmap(
        cache_root / "xy.npy", mode="w+", dtype=np.float32, shape=(total, 2)
    )
    desc_mm = np.lib.format.open_memmap(
        cache_root / "desc.npy", mode="w+", dtype=np.uint8, shape=(total, 32)
    )
    shapes_mm = np.lib.format.open_memmap(
        cache_root / "shapes.npy", mode="w+", dtype=np.int32, shape=(len(ordered), 2)
    )

    point_cursor = 0
    row_cursor = 0
    for chunk_path in chunk_paths:
        data = np.load(chunk_path, allow_pickle=False)
        n_points = len(data["xy"])
        n_rows = len(data["shapes"])
        xy_mm[point_cursor:point_cursor + n_points] = data["xy"]
        desc_mm[point_cursor:point_cursor + n_points] = data["desc"]
        shapes_mm[row_cursor:row_cursor + n_rows] = data["shapes"]
        point_cursor += n_points
        row_cursor += n_rows
        del data
    xy_mm.flush()
    desc_mm.flush()
    shapes_mm.flush()
    del xy_mm, desc_mm, shapes_mm
    shutil.rmtree(chunk_root)

    offsets_mm = np.load(cache_root / "offsets.npy", mmap_mode="r")
    xy_read = np.load(cache_root / "xy.npy", mmap_mode="r")
    desc_read = np.load(cache_root / "desc.npy", mmap_mode="r")
    shapes_read = np.load(cache_root / "shapes.npy", mmap_mode="r")

    if offsets_mm.shape != (len(ordered) + 1,) or shapes_read.shape != (len(ordered), 2):
        raise TrackBError("candidate packed ORB cache shape mismatch")
    if int(offsets_mm[-1]) != len(xy_read) or len(xy_read) != len(desc_read):
        raise TrackBError("candidate packed ORB cache offsets are inconsistent")

    def get(row_id: str):
        index = row_index[row_id]
        lo, hi = int(offsets_mm[index]), int(offsets_mm[index + 1])
        return {
            "xy": np.asarray(xy_read[lo:hi], dtype=np.float32),
            "desc": np.asarray(desc_read[lo:hi], dtype=np.uint8),
            "shape": tuple(int(x) for x in shapes_read[index]),
        }

    return get



def _geometric_verify_pairs(
    pairs,
    *,
    evidence_path: Path,
    get_a,
    get_b,
    cross: bool,
    workers: int,
    audit_policy,
    batch_size: int = 1024,
):
    """Verify candidate pairs with bounded concurrency; persist accepted evidence only plus a full funnel summary."""
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = set()
    accepted_rows = []
    counts = Counter()
    processed = 0
    iterator = iter(pairs)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        while True:
            batch = list(islice(iterator, int(batch_size)))
            if not batch:
                break
            def one(pair):
                a, b = pair
                return pair, verify_orb_pair(get_a(a), get_b(b), policy=audit_policy)
            for (a, b), verdict in pool.map(one, batch):
                processed += 1
                counts[str(verdict.get("decision_stage", "UNKNOWN"))] += 1
                if verdict["accepted"]:
                    accepted.add((a, b))
                    row = ({"external_row_id": a, "historical_id": b} if cross else {"a": a, "b": b})
                    row.update(verdict)
                    accepted_rows.append(row)
            if processed % 50000 < len(batch):
                print(f"Geometric verification {processed:,} pairs; accepted={len(accepted):,}", flush=True)
    sort_keys = (lambda row: (row["external_row_id"], row["historical_id"])) if cross else (lambda row: (row["a"], row["b"]))
    audit_write_jsonl(evidence_path, sorted(accepted_rows, key=sort_keys))
    summary = {
        "generated_nonexact_pair_count": processed,
        "accepted_nonexact_pair_count": len(accepted),
        "decision_stage_counts": dict(sorted(counts.items())),
    }
    return accepted, summary


def _audit_candidate(
    *,
    candidate_id: str,
    bundle,
    historical_bundle,
    core_bundle,
    output_root: Path,
    class_map: dict[str, int],
    mapping: dict[str, str],
    expected_count: int,
    expected_doi: str,
    expected_version: str,
    eligible_subtree: str | None,
    scope: str,
    expected_label_support: dict[str, int] | None,
    workers: int,
    device: str,
    dino_model,
    hist_rows,
    hist_features,
    hist_orb_get,
    audit_policy,
    audit_scratch_root: Path,
) -> dict[str, Any]:
    stage(f"candidate audit :: {candidate_id}")
    candidate_out = output_root / "candidates" / candidate_id
    candidate_out.mkdir(parents=True, exist_ok=False)

    source_ok = (
        str(bundle.manifest.get("doi", "")) == expected_doi
        and str(bundle.manifest.get("version", "")) == expected_version
    )
    source_metadata_hash = None
    source_metadata = {}
    known_relation = False
    unresolved_lineage = True
    try:
        source_metadata_path = resolve_bundle_file(bundle, "source_metadata_record")
        source_metadata = load_json(source_metadata_path)
        source_metadata_hash = sha256_file(source_metadata_path)
        source_ok = source_ok and str(source_metadata.get("doi", "")) == expected_doi
        source_ok = source_ok and str(source_metadata.get("version", "")) == expected_version
        source_ok = source_ok and bool(str(source_metadata.get("license_or_access_text", "")).strip())
        source_ok = source_ok and bool(str(source_metadata.get("retrieved_at", "")).strip())
        source_ok = source_ok and bool(str(source_metadata.get("source_url", "")).strip())
        frozen_lineage = load_json(resolve_bundle_file(core_bundle, "execution_lock")).get("external_lineage_review", {})
        source_ok = source_ok and source_metadata.get("lineage_review_id") == frozen_lineage.get("review_id")
        source_ok = source_ok and source_metadata.get("lineage_review_sha256") == frozen_lineage.get("sha256")
        lineage_status = str(source_metadata.get("lineage_review_status", ""))
        known_relation = source_metadata.get("known_historical_contributor_relationship")
        source_ok = source_ok and lineage_status in {"PASS_NO_KNOWN_RELATIONSHIP", "RESIDUAL_UNCERTAINTY"}
        source_ok = source_ok and isinstance(known_relation, bool)
        unresolved_lineage = bool(known_relation is True or lineage_status != "PASS_NO_KNOWN_RELATIONSHIP")
    except Exception:
        source_ok = False
    data_root = _resolve_dir(bundle, "data_root")
    all_items = discover_candidate_images(data_root, eligible_subtree=eligible_subtree, allowed_labels=None)
    if len(all_items) != int(expected_count):
        source_ok = False
    records, decode_failures = _parallel_records(all_items, data_root, workers)
    observed_label_support: dict[str, int] = {}
    for _path, label in all_items:
        observed_label_support[label] = observed_label_support.get(label, 0) + 1
    if expected_label_support:
        for label, count in expected_label_support.items():
            if int(observed_label_support.get(label, 0)) != int(count):
                source_ok = False
    raw_manifest_path = candidate_out / "raw_manifest.csv"
    _write_candidate_raw_manifest(raw_manifest_path, records, decode_failures)
    audit_write_jsonl(candidate_out / "decode_failures.jsonl", decode_failures)

    mapped_source_labels = set(mapping)
    mapped_records = [r for r in records if r.source_label in mapped_source_labels]
    if candidate_id in {"irish_potato", "gvlid_grape"} and any(label not in mapped_source_labels for _path, label in all_items):
        source_ok = False
    mapping_ok = len(set(mapping.values())) == len(mapping) and all(target in class_map for target in mapping.values())

    # Prediction-blind family construction over all eligible originals.
    exact_within = exact_duplicate_pairs(records)
    phash_within = near_hash_pairs(records, field="phash64", radius=audit_policy.phash_radius)
    dhash_within = near_hash_pairs(records, field="dhash64", radius=audit_policy.dhash_radius)
    image_paths = [data_root / r.relative_path for r in records]
    candidate_features = encode_audit_features(dino_model, image_paths, ctc_v2_eval_transform, device, batch_size=64)
    dino_within = _self_dino_pairs(candidate_features, [r.row_id for r in records], device=device, top_k=audit_policy.dino_top_k)
    candidate_pair_union = phash_within | dhash_within | dino_within | exact_within

    candidate_orb_get = _candidate_orb_getter(
        records,
        data_root,
        workers,
        audit_policy,
        audit_scratch_root / candidate_id / "candidate_orb",
    )
    accepted_within = set(exact_within)
    accepted_near_within, within_geo_summary = _geometric_verify_pairs(
        candidate_pair_union - exact_within,
        evidence_path=candidate_out / "within_geometric_accepts.jsonl",
        get_a=candidate_orb_get, get_b=candidate_orb_get, cross=False, workers=workers,
        audit_policy=audit_policy,
    )
    accepted_within.update(accepted_near_within)
    within_generation_summary = {
        "exact_pair_count": len(exact_within),
        "phash_candidate_pair_count": len(phash_within),
        "dhash_candidate_pair_count": len(dhash_within),
        "dino_candidate_pair_count": len(dino_within),
        "unique_union_pair_count": len(candidate_pair_union),
        **within_geo_summary,
    }
    atomic_write_json(candidate_out / "within_comparison_summary.json", within_generation_summary)

    # Historical comparison is claim-eligible mapped originals only.
    mapped_index = [i for i, r in enumerate(records) if r.source_label in mapped_source_labels]
    mapped_features = candidate_features[mapped_index]
    mapped_order = [records[i] for i in mapped_index]
    exact_cross = _cross_exact_pairs(mapped_order, hist_rows)
    phash_cross = cross_hash_pairs(mapped_order, hist_rows, field="phash64", radius=audit_policy.phash_radius)
    dhash_cross = cross_hash_pairs(mapped_order, hist_rows, field="dhash64", radius=audit_policy.dhash_radius)
    dino_idx, _ = topk_cosine_neighbors(mapped_features, hist_features, k=audit_policy.dino_top_k, device=device)
    dino_cross = {
        (row.row_id, hist_rows[int(j)]["hist_id"])
        for row, neighbor_row in zip(mapped_order, dino_idx)
        for j in neighbor_row
    }
    cross_union = exact_cross | phash_cross | dhash_cross | dino_cross
    hist_id_to_idx = {row["hist_id"]: i for i, row in enumerate(hist_rows)}
    accepted_cross = set(exact_cross)
    def hist_get(hist_id: str):
        return hist_orb_get(hist_id_to_idx[hist_id])
    accepted_near_cross, cross_geo_summary = _geometric_verify_pairs(
        cross_union - exact_cross,
        evidence_path=candidate_out / "historical_geometric_accepts.jsonl",
        get_a=candidate_orb_get, get_b=hist_get, cross=True, workers=workers,
        audit_policy=audit_policy,
    )
    accepted_cross.update(accepted_near_cross)
    historical_generation_summary = {
        "exact_pair_count": len(exact_cross),
        "phash_candidate_pair_count": len(phash_cross),
        "dhash_candidate_pair_count": len(dhash_cross),
        "dino_candidate_pair_count": len(dino_cross),
        "unique_union_pair_count": len(cross_union),
        **cross_geo_summary,
    }
    atomic_write_json(candidate_out / "historical_comparison_summary.json", historical_generation_summary)

    audit_write_jsonl(candidate_out / "accepted_within_edges.jsonl", ({"a": a, "b": b} for a, b in sorted(accepted_within)))
    audit_write_jsonl(candidate_out / "accepted_historical_edges.jsonl", ({"external_row_id": a, "historical_id": b} for a, b in sorted(accepted_cross)))
    components = build_family_components([r.row_id for r in records], accepted_within)
    contaminated_ids = {external_id for external_id, _ in accepted_cross}
    families = representative_manifest(records, components, contaminated_ids=contaminated_ids)
    by_id = {r.row_id: r for r in records}

    representatives = []
    support: dict[str, int] = {source: 0 for source in mapping}
    for family in families:
        rep = by_id[family["representative_row_id"]]
        claim_eligible = (
            not family["label_conflict"]
            and not family["historically_contaminated"]
            and rep.source_label in mapping
        )
        row = {**family, "representative_relative_path": rep.relative_path, "source_label": rep.source_label, "claim_eligible": claim_eligible}
        if claim_eligible:
            row["target_label"] = mapping[rep.source_label]
            row["target_class_index"] = class_map[mapping[rep.source_label]]
            support[rep.source_label] += 1
        representatives.append(row)
    audit_write_jsonl(candidate_out / "families.jsonl", representatives)
    exclusions = []
    for row in representatives:
        reasons = []
        if row["label_conflict"]:
            reasons.append("EXT_LABEL_CONFLICT_FAMILY")
        if row["historically_contaminated"]:
            reasons.append("HISTORICAL_CONTAMINATION")
        if row["source_label"] not in mapping:
            reasons.append("UNMAPPED_SOURCE_LABEL")
        if reasons:
            exclusions.append({
                "family_id": row["family_id"],
                "representative_row_id": row["representative_row_id"],
                "source_label": row["source_label"],
                "reasons": reasons,
            })
    audit_write_jsonl(candidate_out / "exclusions.jsonl", exclusions)

    historical_surface_complete = len(hist_rows) == 117546
    grade = assign_candidate_grade(
        source_identity_ok=source_ok,
        mapping_ok=mapping_ok,
        family_support=support,
        required_labels=mapping.keys(),
        historical_surface_complete=historical_surface_complete,
        accepted_historical_link_count=len(accepted_cross),
        unresolved_lineage=unresolved_lineage,
        known_historical_contributor_relationship=known_relation,
    )

    final_reps = deterministic_representative_order(
        [row for row in representatives if row["claim_eligible"]],
        seed=audit_policy.external_family_order_seed,
    )
    final_rep_path = candidate_out / "sealed_representatives.jsonl"
    audit_write_jsonl(final_rep_path, final_reps)
    source_record = {
        "candidate_id": candidate_id,
        "scope": scope,
        "doi": bundle.manifest.get("doi"),
        "version": bundle.manifest.get("version"),
        "expected_original_count": expected_count,
        "observed_original_count": len(all_items),
        "decoded_image_count": len(records),
        "decode_failure_count": len(decode_failures),
        "observed_source_label_support": observed_label_support,
        "expected_source_label_support": expected_label_support,
        "source_identity_ok": source_ok,
        "source_metadata_record_sha256": source_metadata_hash,
        "source_url": source_metadata.get("source_url"),
        "retrieved_at": source_metadata.get("retrieved_at"),
        "license_or_access_text": source_metadata.get("license_or_access_text"),
        "lineage_review_status": source_metadata.get("lineage_review_status"),
        "known_historical_contributor_relationship": source_metadata.get("known_historical_contributor_relationship"),
        "unresolved_lineage": unresolved_lineage,
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "mapping": mapping,
        "mapping_sha256": sha256_json(mapping),
        "family_support": support,
        "family_support_cropcop": {mapping[source]: int(count) for source, count in support.items()},
        "accepted_historical_link_count": len(accepted_cross),
        "unresolved_lineage": unresolved_lineage,
        "grade_reasons": list(grade.reasons),
        "historical_surface_complete": historical_surface_complete,
        "grade": grade.grade,
        "claim_mode": grade.claim_mode,
        "grade_reasons": list(grade.reasons),
    }
    atomic_write_json(candidate_out / "audit_summary.json", source_record)

    seal_payload = {
        "schema_version": "1.0",
        "authority_id": AUTHORITY_ID,
        "downstream_authority_sha256": sha256_file(resolve_bundle_file(core_bundle, "downstream_authority")),
        "execution_lock_sha256": sha256_file(resolve_bundle_file(core_bundle, "execution_lock")),
        "candidate_id": candidate_id,
        "scope": scope,
        "grade": grade.grade,
        "claim_mode": grade.claim_mode,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_doi": str(bundle.manifest.get("doi", "")),
        "source_version": str(bundle.manifest.get("version", "")),
        "source_identity_ok": bool(source_ok),
        "mapping_ok": bool(mapping_ok),
        "unresolved_lineage": bool(unresolved_lineage),
        "known_historical_contributor_relationship": bool(known_relation),
        "grade_reasons": list(grade.reasons),
        "mapping": mapping,
        "candidate_input_manifest_sha256": sha256_file(bundle.manifest_path),
        "source_manifest_sha256": sha256_file(raw_manifest_path),
        "source_metadata_record_sha256": source_metadata_hash,
        "decode_failure_ledger_sha256": sha256_file(candidate_out / "decode_failures.jsonl"),
        "mapping_sha256": sha256_json(mapping),
        "family_graph_sha256": sha256_file(candidate_out / "families.jsonl"),
        "representative_manifest_sha256": sha256_file(final_rep_path),
        "exclusion_ledger_sha256": sha256_file(candidate_out / "exclusions.jsonl"),
        "historical_compare_input_manifest_sha256": sha256_file(historical_bundle.manifest_path),
        "historical_geometric_accepts_sha256": sha256_file(candidate_out / "historical_geometric_accepts.jsonl"),
        "historical_comparison_summary_sha256": sha256_file(candidate_out / "historical_comparison_summary.json"),
        "within_geometric_accepts_sha256": sha256_file(candidate_out / "within_geometric_accepts.jsonl"),
        "within_comparison_summary_sha256": sha256_file(candidate_out / "within_comparison_summary.json"),
        "accepted_within_edges_sha256": sha256_file(candidate_out / "accepted_within_edges.jsonl"),
        "accepted_historical_edges_sha256": sha256_file(candidate_out / "accepted_historical_edges.jsonl"),
        "family_support": support,
        "accepted_historical_link_count": len(accepted_cross),
        "historical_surface_complete": historical_surface_complete,
        "authorized_r07_checkpoint_sha256": R07_CHECKPOINTS if grade.grade in {"EXT-I", "EXT-S"} else {},
        "class_map_sha256": CLASS_MAP_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "preprocessing": {
            "id": "CTC-v2-eval",
            "entrypoint": "cropcop_je.data.ctc_v2_eval_transform",
            "config_sha256": "53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4",
            "source_commit": "604aafd51e20e70098ce4af647e90c8ff558a9e8"
        },
        "bootstrap_seed": audit_policy.bootstrap_seed,
        "bootstrap_replicates": audit_policy.bootstrap_replicates,
        "external_family_order_seed": audit_policy.external_family_order_seed,
        "prediction_count_at_seal": 0,
    }
    seal = build_candidate_seal(seal_payload)
    atomic_write_json(candidate_out / "seal.json", seal)
    verify_candidate_seal(seal)
    return {"candidate_id": candidate_id, "root": candidate_out, "grade": grade.grade, "seal": seal, "representatives": final_reps}


def _preflight(core, historical, output_root: Path, device: str):
    stage("0 :: authority, environment, and R07 family replay")
    authority = _json_file(core, "downstream_authority")
    execution_lock = _json_file(core, "execution_lock")
    validate_downstream_authority(authority)
    validate_execution_lock(execution_lock)
    code_attestation_path = resolve_bundle_file(core, "code_attestation")
    expected_attestation_sha = str(execution_lock.get("code_attestation_sha256", ""))
    if sha256_file(code_attestation_path) != expected_attestation_sha:
        raise TrackBError("core code-attestation bytes differ from the execution lock")
    repo_root = _resolve_dir(core, "repository_root")
    verify_code_attestation(repo_root, code_attestation_path)

    import numpy as np
    import PIL
    import torch
    import torchvision
    import timm
    import transformers
    import huggingface_hub
    import safetensors
    try:
        import cv2
    except Exception as exc:
        raise TrackBError(f"OpenCV unavailable: {exc}") from exc
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__.split("+", 1)[0],
        "torchvision": torchvision.__version__.split("+", 1)[0],
        "timm": timm.__version__,
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "kaggle": importlib.metadata.version("kaggle"),
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    expected = execution_lock["software"]
    if str(environment["python"]) != str(expected["python"]):
        raise TrackBError(
            f"locked Python drift: expected {expected['python']}, got {environment['python']}"
        )
    for key in ("torch", "torchvision", "timm", "numpy", "Pillow", "kaggle", "transformers", "huggingface_hub", "safetensors"):
        if str(environment[key]) != str(expected[key]):
            raise TrackBError(f"locked dependency drift: {key}: expected {expected[key]}, got {environment[key]}")
    if str(environment["opencv"]) != "4.13.0":
        raise TrackBError(f"locked OpenCV drift: expected 4.13.0 runtime, got {environment['opencv']}")
    cv2.setNumThreads(1)
    environment["opencv_threads"] = int(cv2.getNumThreads())
    if not torch.cuda.is_available():
        raise TrackBError("Track B claim run requires CUDA")
    environment["code_attestation_sha256"] = expected_attestation_sha
    environment["repository_root"] = str(repo_root)
    atomic_write_json(output_root / "environment.json", environment)

    class_map_path = resolve_bundle_file(core, "class_map")
    manifest_path = resolve_bundle_file(core, "v1_manifest")
    class_map = _class_map(class_map_path)
    if sha256_file(manifest_path) != DATASET_MANIFEST_SHA256:
        raise TrackBError("V1 manifest identity mismatch")
    val_root = _resolve_dir(core, "v1_validation_root")
    if not (val_root / "val").is_dir():
        raise TrackBError("core V1 validation surface does not expose the required val/ subtree")
    forbidden_test_dirs = [p for p in val_root.iterdir() if p.is_dir() and p.name.lower() in {"test", "test_consumed", "v1_test", "ds-v1-test-consumed"}]
    if forbidden_test_dirs:
        raise TrackBError("consumed V1-test bytes are present beside the validation surface: " + ", ".join(p.name for p in forbidden_test_dirs))
    val_rows = load_frozen_v1_rows(
        manifest_path,
        class_map_path,
        expected_manifest_sha256=DATASET_MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        expected_count=VAL_COUNT,
    )

    if sha256_file(resolve_bundle_file(core, "dino_factory_manifest")) != DINO_FACTORY_MANIFEST_SHA256:
        raise TrackBError("core DINO factory-manifest identity mismatch")

    replay = {}
    for seed in ("S1", "S2", "S3"):
        model, _payload = load_r07_checkpoint(resolve_bundle_file(core, f"r07_{seed.lower()}"), seed_label=seed)
        run_record_path = resolve_bundle_file(core, f"r07_{seed.lower()}_run_record")
        if sha256_file(run_record_path) != R07_RUN_RECORDS[seed]["sha256"]:
            raise TrackBError(f"R07 {seed} authoritative replay-record SHA mismatch")
        run_record = load_json(run_record_path)
        if str(run_record.get("run_id", "")) != R07_RUN_RECORDS[seed]["run_id"]:
            raise TrackBError(f"R07 {seed} authoritative replay-record ID mismatch")
        selected_sha = (
            ((run_record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
            or (run_record.get("result_summary") or {}).get("selected_checkpoint_sha256")
            or run_record.get("selected_checkpoint_sha256")
        )
        if str(selected_sha) != R07_CHECKPOINTS[seed]:
            raise TrackBError(f"R07 {seed} replay record/checkpoint identity mismatch")
        expected_metrics = run_record.get("result_summary", {}).get("selected_metrics") or {}
        predictions, summary = clean_replay(model, val_rows, val_root, torch.device(device), batch_size=32, num_workers=4)
        gate = validate_selected_checkpoint_replay(summary, expected_metrics)
        if gate["status"] != "PASS":
            raise TrackBError(f"R07 {seed} frozen validation replay failed: {gate}")
        replay[seed] = {"gate": gate, "summary": summary, "prediction_digest": sha256_json(predictions)}
        del model
        torch.cuda.empty_cache()
    atomic_write_json(output_root / "r07_family_replay.json", replay)

    hist_manifest = resolve_bundle_file(historical, "historical_manifest")
    hist_rows = _read_hist_manifest(hist_manifest)
    declared_hist_count = int(historical.manifest.get("image_count", -1))
    if declared_hist_count != len(hist_rows):
        raise TrackBError(f"historical comparison manifest declares {declared_hist_count} rows but contains {len(hist_rows)}")
    if declared_hist_count == 92744:
        expected_partial = {
            "coverage_scope": "V1_TRAIN_VAL_ONLY",
            "ext_i_eligible": False,
            "maximum_evidence_grade": "EXT-S",
            "v1_test_image_bytes_accessed": False,
        }
        for key, value in expected_partial.items():
            if historical.manifest.get(key) != value:
                raise TrackBError(f"safe historical comparison package policy mismatch: {key}")
    elif declared_hist_count == 117546:
        raise TrackBError(
            "full 117,546-image EXT-I route is dormant until a recovered pre-test comparison "
            "representation receives a formal provenance binding before candidate audit; row count alone "
            "cannot authorize reopening EXT-I"
        )
    else:
        raise TrackBError(
            f"unsupported historical comparison surface size {declared_hist_count}; "
            "current executable authority accepts exactly the safe 92,744-image V1 train+validation surface"
        )
    import numpy as np
    hist_features = np.load(resolve_bundle_file(historical, "dino_features"), mmap_mode="r")
    if hist_features.shape != (len(hist_rows), 768):
        raise TrackBError(f"historical DINO feature surface must be ({len(hist_rows)}, 768), got {hist_features.shape}")
    if historical.manifest.get("dino_audit_encoder_sha256") != DINO_AUDIT_SHA256:
        raise TrackBError("historical feature package is not bound to the frozen DINO audit encoder")
    if historical.manifest.get("code_attestation_sha256") != expected_attestation_sha:
        raise TrackBError("historical comparison package was not built under the locked Track-B code attestation")
    if historical.manifest.get("features_l2_normalized") is not True:
        raise TrackBError("historical DINO feature package does not assert L2-normalized rows")
    if historical.manifest.get("ctc_v2_config_sha256") != "53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4":
        raise TrackBError("historical DINO feature preprocessing binding mismatch")
    for start in range(0, len(hist_features), 8192):
        block = np.asarray(hist_features[start:start + 8192], dtype=np.float32)
        if not np.isfinite(block).all():
            raise TrackBError("historical DINO feature package contains non-finite values")
        norms = np.linalg.norm(block, axis=1)
        if not np.all(np.abs(norms - 1.0) <= 1e-4):
            raise TrackBError("historical DINO feature package has non-normalized rows")
    hist_orb_get = _load_hist_orb(historical, hist_rows)
    return authority, execution_lock, class_map, hist_rows, hist_features, hist_orb_get


def _protected_inference(candidate, core, class_map, output_root: Path, device: str, audit_policy):
    if candidate["grade"] not in {"EXT-I", "EXT-S"}:
        return None
    stage(f"5 :: protected R07 inference :: {candidate['candidate_id']}")
    import torch

    seal = candidate["seal"]
    verify_candidate_seal(seal)
    rep_rows = candidate["representatives"]
    mapped_indices = sorted({int(row["target_class_index"]) for row in rep_rows})
    source_root = candidate["root"]
    # candidate root stores evidence; raw images remain in their Kaggle bundle, so preserve the path in each row.
    # Caller injects absolute_image_path before this stage.
    prediction_sets = {}
    seed_metrics = {}
    for seed in ("S1", "S2", "S3"):
        model, _ = load_r07_checkpoint(resolve_bundle_file(core, f"r07_{seed.lower()}"), seed_label=seed)
        class _Dataset:
            def __len__(self): return len(rep_rows)
            def __getitem__(self, index):
                from PIL import Image
                row = rep_rows[int(index)]
                with Image.open(row["absolute_image_path"]) as image:
                    x = ctc_v2_eval_transform(image)
                return x, int(row["target_class_index"]), str(row["representative_row_id"])
        predictions = prediction_rows(model, _Dataset(), torch.device(device), batch_size=64, num_workers=4)
        prediction_sets[seed] = predictions
        seed_metrics[seed] = mapped_scope_metrics(predictions, mapped_indices)
        out = output_root / "inference" / candidate["candidate_id"] / seed
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "predictions.jsonl", predictions)
        atomic_write_json(out / "metrics.json", seed_metrics[seed])
        del model
        torch.cuda.empty_cache()
    validate_same_prediction_surface(prediction_sets)
    summary = three_seed_summary(seed_metrics)
    bootstrap = bootstrap_three_seed_macro_f1(
        prediction_sets,
        mapped_indices,
        replicates=audit_policy.bootstrap_replicates,
        seed=audit_policy.bootstrap_seed,
    )
    analysis_dir = output_root / "analysis" / candidate["candidate_id"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(analysis_dir / "seed_metrics.json", seed_metrics)
    atomic_write_json(analysis_dir / "three_seed_summary.json", summary)
    atomic_write_json(analysis_dir / "bootstrap.json", bootstrap)
    return {"seed_metrics": seed_metrics, "summary": summary, "bootstrap": bootstrap, "prediction_sets": prediction_sets}


def _inject_image_paths(candidate, candidate_bundle):
    data_root = _resolve_dir(candidate_bundle, "data_root")
    for row in candidate["representatives"]:
        path = (data_root / row["representative_relative_path"]).resolve()
        if data_root not in path.parents and path != data_root:
            raise TrackBError("sealed representative path escapes candidate root")
        row["absolute_image_path"] = str(path)



def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _independent_candidate_qa(candidate, output_root: Path, audit_policy) -> dict[str, Any]:
    cid = candidate["candidate_id"]
    seal_path = candidate["root"] / "seal.json"
    seal = load_json(seal_path)
    verify_candidate_seal(seal)
    rep_path = candidate["root"] / "sealed_representatives.jsonl"
    if sha256_file(rep_path) != seal.get("representative_manifest_sha256"):
        raise TrackBError(f"{cid}: sealed representative manifest changed after seal")
    family_path = candidate["root"] / "families.jsonl"
    if sha256_file(family_path) != seal.get("family_graph_sha256"):
        raise TrackBError(f"{cid}: family graph changed after seal")
    accepted_hist = candidate["root"] / "accepted_historical_edges.jsonl"
    if sha256_file(accepted_hist) != seal.get("accepted_historical_edges_sha256"):
        raise TrackBError(f"{cid}: historical accepted-edge ledger changed after seal")
    exclusion_path = candidate["root"] / "exclusions.jsonl"
    if sha256_file(exclusion_path) != seal.get("exclusion_ledger_sha256"):
        raise TrackBError(f"{cid}: exclusion ledger changed after seal")
    decode_path = candidate["root"] / "decode_failures.jsonl"
    if sha256_file(decode_path) != seal.get("decode_failure_ledger_sha256"):
        raise TrackBError(f"{cid}: decode-failure ledger changed after seal")
    accepted_hist_rows = _read_jsonl(accepted_hist)
    if len(accepted_hist_rows) != int(seal.get("accepted_historical_link_count", -1)):
        raise TrackBError(f"{cid}: historical accepted-edge count mismatch")
    if seal.get("grade") == "EXT-I" and accepted_hist_rows:
        raise TrackBError(f"{cid}: EXT-I candidate has accepted historical links")
    reps = _read_jsonl(rep_path)
    support = {}
    for row in reps:
        label = str(row["source_label"])
        support[label] = support.get(label, 0) + 1
    if support != {str(k): int(v) for k, v in seal.get("family_support", {}).items()}:
        raise TrackBError(f"{cid}: sealed family-support table does not recompute")
    result = {
        "candidate_id": cid,
        "grade": seal["grade"],
        "seal_sha256": seal["seal_sha256"],
        "representative_count": len(reps),
        "family_support": support,
        "historical_link_count": len(accepted_hist_rows),
    }
    if seal["grade"] in {"EXT-I", "EXT-S"}:
        seed_rows = {}
        persisted_seed_metrics = {}
        mapped_indices = sorted({int(row["target_class_index"]) for row in reps})
        for seed in ("S1", "S2", "S3"):
            pred_path = output_root / "inference" / cid / seed / "predictions.jsonl"
            metric_path = output_root / "inference" / cid / seed / "metrics.json"
            seed_rows[seed] = _read_jsonl(pred_path)
            persisted_seed_metrics[seed] = load_json(metric_path)
            recomputed = mapped_scope_metrics(seed_rows[seed], mapped_indices)
            if sha256_json(recomputed) != sha256_json(persisted_seed_metrics[seed]):
                raise TrackBError(f"{cid}/{seed}: persisted metrics do not independently recompute")
        validate_same_prediction_surface(seed_rows)
        summary = three_seed_summary(persisted_seed_metrics)
        persisted_summary = load_json(output_root / "analysis" / cid / "three_seed_summary.json")
        if sha256_json(summary) != sha256_json(persisted_summary):
            raise TrackBError(f"{cid}: three-seed summary does not independently recompute")
        bootstrap = bootstrap_three_seed_macro_f1(
            seed_rows,
            mapped_indices,
            replicates=audit_policy.bootstrap_replicates,
            seed=audit_policy.bootstrap_seed,
        )
        persisted_bootstrap = load_json(output_root / "analysis" / cid / "bootstrap.json")
        if sha256_json(bootstrap) != sha256_json(persisted_bootstrap):
            raise TrackBError(f"{cid}: bootstrap does not independently reproduce")
        result["three_seed_mean_macro_f1"] = summary["mean_macro_f1"]
        result["three_seed_sample_sd_macro_f1"] = summary["sample_sd_macro_f1"]
        result["bootstrap_ci95"] = bootstrap["three_seed"]["bootstrap_mean_macro_f1_ci95_percentile"]
    return result


def _science_preimage_manifest(output_root: Path, core, candidates) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "authority_id": AUTHORITY_ID,
        "downstream_authority_sha256": sha256_file(resolve_bundle_file(core, "downstream_authority")),
        "execution_lock_sha256": sha256_file(resolve_bundle_file(core, "execution_lock")),
        "code_attestation_sha256": sha256_file(resolve_bundle_file(core, "code_attestation")),
        "class_map_sha256": CLASS_MAP_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "authorized_r07_checkpoint_sha256": R07_CHECKPOINTS,
        "candidates": {},
    }
    for candidate in candidates:
        root = candidate["root"]
        seal = candidate["seal"]
        payload["candidates"][candidate["candidate_id"]] = {
            "candidate_id": candidate["candidate_id"],
            "source_doi": seal["source_doi"],
            "source_version": seal["source_version"],
            "grade": seal["grade"],
            "claim_mode": seal["claim_mode"],
            "mapping_sha256": seal["mapping_sha256"],
            "family_support": seal["family_support"],
            "accepted_historical_link_count": seal["accepted_historical_link_count"],
            "source_manifest_sha256": sha256_file(root / "raw_manifest.csv"),
            "family_graph_sha256": sha256_file(root / "families.jsonl"),
            "representative_manifest_sha256": sha256_file(root / "sealed_representatives.jsonl"),
            "accepted_within_edges_sha256": sha256_file(root / "accepted_within_edges.jsonl"),
            "accepted_historical_edges_sha256": sha256_file(root / "accepted_historical_edges.jsonl"),
        }
    payload["science_preimage_sha256"] = sha256_json(payload)
    return payload


def _stable_science_manifest(output_root: Path, core, candidates) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "authority_id": AUTHORITY_ID,
        "downstream_authority_sha256": sha256_file(resolve_bundle_file(core, "downstream_authority")),
        "execution_lock_sha256": sha256_file(resolve_bundle_file(core, "execution_lock")),
        "class_map_sha256": CLASS_MAP_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "authorized_r07_checkpoint_sha256": R07_CHECKPOINTS,
        "candidates": {},
    }
    for candidate in candidates:
        cid = candidate["candidate_id"]
        root = candidate["root"]
        seal = candidate["seal"]
        row = {
            "candidate_id": cid,
            "source_doi": seal["source_doi"],
            "source_version": seal["source_version"],
            "grade": seal["grade"],
            "claim_mode": seal["claim_mode"],
            "mapping_sha256": seal["mapping_sha256"],
            "family_support": seal["family_support"],
            "accepted_historical_link_count": seal["accepted_historical_link_count"],
            "source_manifest_sha256": sha256_file(root / "raw_manifest.csv"),
            "family_graph_sha256": sha256_file(root / "families.jsonl"),
            "representative_manifest_sha256": sha256_file(root / "sealed_representatives.jsonl"),
            "exclusion_ledger_sha256": sha256_file(root / "exclusions.jsonl"),
            "accepted_within_edges_sha256": sha256_file(root / "accepted_within_edges.jsonl"),
            "accepted_historical_edges_sha256": sha256_file(root / "accepted_historical_edges.jsonl"),
            "prediction_sha256": {},
            "metric_sha256": {},
            "analysis_sha256": {},
        }
        if seal["grade"] in {"EXT-I", "EXT-S"}:
            for seed in ("S1", "S2", "S3"):
                row["prediction_sha256"][seed] = sha256_file(
                    output_root / "inference" / cid / seed / "predictions.jsonl"
                )
                row["metric_sha256"][seed] = sha256_file(
                    output_root / "inference" / cid / seed / "metrics.json"
                )
            for name in ("seed_metrics.json", "three_seed_summary.json", "bootstrap.json"):
                row["analysis_sha256"][name] = sha256_file(output_root / "analysis" / cid / name)
        payload["candidates"][cid] = row
    payload["trackb_science_sha256"] = sha256_json(payload)
    atomic_write_json(output_root / "TRACKB_SCIENCE_MANIFEST.json", payload)
    return payload


def _verify_zip_archive(path: Path) -> dict[str, Any]:
    import hashlib
    import zipfile

    members = {}
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise TrackBError(f"evidence ZIP CRC verification failed at member: {bad}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            h = hashlib.sha256()
            with zf.open(info, "r") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            members[info.filename] = {
                "bytes": info.file_size,
                "sha256": h.hexdigest(),
            }
    if not members:
        raise TrackBError(f"evidence ZIP is empty: {path}")
    return {
        "member_count": len(members),
        "members_sha256": sha256_json(members),
    }


def _package_outputs(output_root: Path) -> dict[str, Any]:
    import zipfile
    complete = output_root / "TRACKB_COMPLETE_EVIDENCE.zip"
    public = output_root / "TRACKB_PUBLIC_EVIDENCE.zip"
    exclude = {complete.name, public.name, "TRACKB_FINAL_CLOSURE.json", "TRACKB_PACKAGE_MANIFEST.json"}
    with zipfile.ZipFile(complete, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path.name not in exclude:
                zf.write(path, path.relative_to(output_root).as_posix())
    public_allow = []
    for path in [
        output_root / "environment.json",
        output_root / "r07_family_replay.json",
        output_root / "dino_audit_encoder_identity.json",
        output_root / "TRACKB_PREDICTION_FIREWALL.json",
        output_root / "TRACKB_FINAL_QA.json",
        output_root / "TRACKB_SCIENCE_MANIFEST.json",
    ]:
        if path.is_file():
            public_allow.append(path)
    for candidate_dir in sorted((output_root / "candidates").glob("*")) if (output_root / "candidates").exists() else []:
        for name in ("audit_summary.json", "seal.json"):
            path = candidate_dir / name
            if path.is_file():
                public_allow.append(path)
    for analysis_dir in sorted((output_root / "analysis").glob("*")) if (output_root / "analysis").exists() else []:
        for name in ("seed_metrics.json", "three_seed_summary.json", "bootstrap.json"):
            path = analysis_dir / name
            if path.is_file():
                public_allow.append(path)
    with zipfile.ZipFile(public, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in public_allow:
            zf.write(path, path.relative_to(output_root).as_posix())

    complete_verification = _verify_zip_archive(complete)
    public_verification = _verify_zip_archive(public)
    return {
        "schema_version": "2.0",
        "status": "PASS",
        "closure_written_after_package_verification": True,
        "complete_evidence_zip": {
            "bytes": complete.stat().st_size,
            "sha256": sha256_file(complete),
            **complete_verification,
        },
        "public_evidence_zip": {
            "bytes": public.stat().st_size,
            "sha256": sha256_file(public),
            **public_verification,
        },
    }

def main() -> int:
    ap = argparse.ArgumentParser(description="CropCop Track B R07 end-to-end external-validation runner")
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--output-root", default="/kaggle/working/trackb_r07")
    ap.add_argument("--scratch-root", default="/kaggle/tmp/cropcop_trackb_r07_audit")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=["preflight", "all"], default="all")
    ap.add_argument("--source-git-sha", default="")
    ap.add_argument("--attempt-dataset-slug", default="")
    ap.add_argument("--attempt-id", default="")
    args = ap.parse_args()

    output_root = Path(args.output_root).resolve()
    audit_scratch_root = Path(args.scratch_root).resolve()
    if audit_scratch_root.exists():
        shutil.rmtree(audit_scratch_root)
    audit_scratch_root.mkdir(parents=True, exist_ok=False)
    if output_root.exists() and any(output_root.iterdir()):
        raise TrackBError(f"output root must be empty for a clean claim run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    inputs = discover_kaggle_inputs(args.input_root)
    core = inputs["core"]
    historical = inputs["historical_compare"]
    authority, lock, class_map, hist_rows, hist_features, hist_orb_get = _preflight(core, historical, output_root, args.device)
    audit_policy = audit_policy_from_lock(lock)
    if args.mode == "preflight":
        atomic_write_json(output_root / "PREFLIGHT_PASS.json", {"status": "PASS", "authority_id": AUTHORITY_ID})
        return 0

    stage("1-4 :: prediction-blind source verification, family audit, grade, and seal")
    dino_model, dino_identity = _load_dino(core)
    atomic_write_json(output_root / "dino_audit_encoder_identity.json", dino_identity)

    grape_mapping = lock["candidate_a"]["mapping"]
    potato_mapping = lock["candidate_b"]["mapping"]
    grape = _audit_candidate(
        candidate_id="gvlid_grape",
        bundle=inputs["gvlid_v5"], historical_bundle=historical, core_bundle=core,
        output_root=output_root, class_map=class_map, mapping=grape_mapping,
        expected_count=3477, expected_doi="10.17632/wkymf8bhcg.5", expected_version="5",
        eligible_subtree=None, scope=lock["candidate_a"]["scope"],
        expected_label_support=None,
        workers=args.workers, device=args.device,
        dino_model=dino_model, hist_rows=hist_rows, hist_features=hist_features, hist_orb_get=hist_orb_get,
        audit_policy=audit_policy,
        audit_scratch_root=audit_scratch_root,
    )
    potato = _audit_candidate(
        candidate_id="irish_potato",
        bundle=inputs["irish_potato"], historical_bundle=historical, core_bundle=core,
        output_root=output_root, class_map=class_map, mapping=potato_mapping,
        expected_count=58709, expected_doi="10.5281/zenodo.8286529", expected_version="01",
        eligible_subtree=None, scope=lock["candidate_b"]["scope"],
        expected_label_support=lock["candidate_b"].get("expected_source_support"),
        workers=args.workers, device=args.device,
        dino_model=dino_model, hist_rows=hist_rows, hist_features=hist_features, hist_orb_get=hist_orb_get,
        audit_policy=audit_policy,
        audit_scratch_root=audit_scratch_root,
    )
    del dino_model
    import torch
    torch.cuda.empty_cache()

    stage("4 :: prediction firewall")
    for candidate in (grape, potato):
        verify_candidate_seal(candidate["seal"])
    firewall = {
        "status": "PASS",
        "authority_id": AUTHORITY_ID,
        "downstream_authority_sha256": sha256_file(resolve_bundle_file(core, "downstream_authority")),
        "execution_lock_sha256": sha256_file(resolve_bundle_file(core, "execution_lock")),
        "candidate_terminal_grades": {grape["candidate_id"]: grape["grade"], potato["candidate_id"]: potato["grade"]},
        "candidate_seal_sha256": {grape["candidate_id"]: grape["seal"]["seal_sha256"], potato["candidate_id"]: potato["seal"]["seal_sha256"]},
        "external_predictions_before_this_attempt_firewall": 0,
        "prediction_firewall_scope": "CURRENT_EXECUTION_ATTEMPT_ONLY",
        "v1_test_accessed": False,
        "all_three_r07_checkpoints_bound": True,
    }
    atomic_write_json(output_root / "TRACKB_PREDICTION_FIREWALL.json", firewall)

    claim_candidates = [candidate for candidate in (grape, potato) if candidate["grade"] in {"EXT-I", "EXT-S"}]
    attempt_state = None
    if claim_candidates:
        if not args.attempt_dataset_slug or not args.attempt_id or len(str(args.source_git_sha)) != 40:
            raise TrackBError(
                "protected inference requires --attempt-dataset-slug, --attempt-id, and exact --source-git-sha"
            )
        preimage = _science_preimage_manifest(output_root, core, (grape, potato))
        previous = read_latest_attempt_state(args.attempt_dataset_slug)
        rerun_gate = validate_prior_attempt_for_rerun(
            previous,
            current_science_preimage_sha256=preimage["science_preimage_sha256"],
        )
        attempt_state = {
            "schema_version": "1.0",
            "attempt_id": str(args.attempt_id),
            "status": "PROTECTED_INFERENCE_STARTED",
            "protected_inference_ever": True,
            "started_at_utc": utc_now(),
            "source_git_sha": str(args.source_git_sha),
            "science_preimage_sha256": preimage["science_preimage_sha256"],
            "science_preimage": preimage,
            "parent_attempt_id": rerun_gate["parent_attempt_id"],
            "prior_attempt_with_protected_inference": rerun_gate["prior_attempt_with_protected_inference"],
        }
        atomic_write_json(output_root / "TRACKB_ATTEMPT_STATE.json", attempt_state)
        attempt_receipt = publish_attempt_state(args.attempt_dataset_slug, attempt_state)
        atomic_write_json(output_root / "TRACKB_ATTEMPT_PUBLICATION.json", attempt_receipt)

    _inject_image_paths(grape, inputs["gvlid_v5"])
    _inject_image_paths(potato, inputs["irish_potato"])
    results = {
        "gvlid_grape": _protected_inference(grape, core, class_map, output_root, args.device, audit_policy),
        "irish_potato": _protected_inference(potato, core, class_map, output_root, args.device, audit_policy),
    }

    stage("7 :: independent closure QA")
    qa = {
        "schema_version": "1.0",
        "status": "PASS",
        "candidate_qa": {},
        "v1_test_accessed": False,
        "new_training_performed": False,
        "external_predictions_before_candidate_seal": False,
    }
    for candidate in (grape, potato):
        qa["candidate_qa"][candidate["candidate_id"]] = _independent_candidate_qa(candidate, output_root, audit_policy)
    qa["qa_sha256"] = sha256_json({k: v for k, v in qa.items() if k != "qa_sha256"})
    atomic_write_json(output_root / "TRACKB_FINAL_QA.json", qa)

    science_manifest = _stable_science_manifest(output_root, core, (grape, potato))
    if attempt_state is not None:
        attempt_state = {
            **attempt_state,
            "status": "SCIENCE_QA_PASS",
            "science_qa_pass_at_utc": utc_now(),
            "trackb_science_sha256": science_manifest["trackb_science_sha256"],
            "final_qa_sha256": qa["qa_sha256"],
        }
        atomic_write_json(output_root / "TRACKB_ATTEMPT_STATE.json", attempt_state)
    packages = _package_outputs(output_root)
    atomic_write_json(output_root / "TRACKB_PACKAGE_MANIFEST.json", packages)

    closure = {
        "schema_version": "2.0",
        "status": "TRACK_B_CLOSED",
        "authority_id": AUTHORITY_ID,
        "elapsed_seconds": time.time() - started,
        "candidate_status": qa["candidate_qa"],
        "final_qa_sha256": qa["qa_sha256"],
        "trackb_science_sha256": science_manifest["trackb_science_sha256"],
        "science_manifest_sha256": sha256_file(output_root / "TRACKB_SCIENCE_MANIFEST.json"),
        "package_manifest_sha256": sha256_file(output_root / "TRACKB_PACKAGE_MANIFEST.json"),
        "package_verification_status": packages["status"],
        "v1_test_accessed": False,
        "new_training_performed": False,
        "external_predictions_before_candidate_seal": False,
    }
    closure["closure_sha256"] = sha256_json({k: v for k, v in closure.items() if k != "closure_sha256"})
    atomic_write_json(output_root / "TRACKB_FINAL_CLOSURE.json", closure)
    print(json.dumps({**closure, "packages": packages}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
