from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import ctc_v2_eval_transform
from cropcop_je.hashing import sha256_file
from cropcop_je.models import load_exact_teacher
from cropcop_je.trackb_r07 import CTC_V2_CONFIG_SHA256, DINO_AUDIT_SHA256, TrackBError, load_json, validate_execution_lock, verify_code_attestation
from cropcop_je.trackb_r07_audit import make_image_record_and_orb

EXPECTED_ROWS = 117546
FEATURE_WIDTH = 768


def _read_source_manifest(path: Path, *, id_col: str, path_col: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    ids: set[str] = set()
    rels: set[str] = set()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {id_col, path_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise TrackBError(f"historical source manifest missing columns: {sorted(missing)}")
        for raw in reader:
            hist_id = str(raw[id_col]).strip()
            rel = str(raw[path_col]).strip().replace("\\", "/")
            posix = PurePosixPath(rel)
            if not hist_id or hist_id in ids:
                raise TrackBError(f"historical source manifest has missing/duplicate ID: {hist_id!r}")
            if not rel or posix.is_absolute() or ".." in posix.parts or rel in rels:
                raise TrackBError(f"historical source manifest has unsafe/duplicate path: {rel!r}")
            ids.add(hist_id); rels.add(rel); rows.append((hist_id, rel))
    if len(rows) != EXPECTED_ROWS:
        raise TrackBError(f"historical source manifest must have exactly {EXPECTED_ROWS} rows, got {len(rows)}")
    return rows


def _load_dino(args):
    factory = json.loads(Path(args.factory_manifest).read_text(encoding="utf-8"))
    entrypoint = str(factory.get("entrypoint", ""))
    if not entrypoint or ":" not in entrypoint:
        raise TrackBError("DINO factory manifest does not bind a valid entrypoint")
    model, identity = load_exact_teacher(
        args.dino_checkpoint,
        factory_spec=entrypoint,
        factory_bundle_manifest=args.factory_manifest,
        repo_root=args.factory_source_root,
        factory_source_root=args.factory_source_root,
    )
    return model, identity


def _pack_orb(rows, image_root: Path, out: Path, *, workers: int, chunk_size: int):
    import numpy as np

    temp = out / ".orb_chunks"
    temp.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "historical_manifest.csv"
    offsets = [0]
    chunk_paths: list[Path] = []
    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["hist_id", "raw_sha256", "phash64", "dhash64", "width", "height"])
        writer.writeheader()
        for chunk_index, start in enumerate(range(0, len(rows), chunk_size)):
            subset = rows[start:start + chunk_size]
            def process(item):
                hist_id, rel = item
                path = (image_root / rel).resolve()
                if image_root not in path.parents and path != image_root:
                    raise TrackBError(f"historical image path escapes root: {rel}")
                if not path.is_file():
                    raise FileNotFoundError(f"historical image missing: {path}")
                rec, orb = make_image_record_and_orb(path, image_root)
                return hist_id, rec, orb
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                computed = list(pool.map(process, subset))
            shapes = np.asarray([row[2]["shape"] for row in computed], dtype=np.int32)
            counts = np.asarray([len(row[2]["desc"]) for row in computed], dtype=np.int32)
            xy = np.concatenate([row[2]["xy"] for row in computed], axis=0) if counts.sum() else np.empty((0, 2), np.float32)
            desc = np.concatenate([row[2]["desc"] for row in computed], axis=0) if counts.sum() else np.empty((0, 32), np.uint8)
            cp = temp / f"chunk_{chunk_index:05d}.npz"
            np.savez(cp, shapes=shapes, counts=counts, xy=xy.astype(np.float32, copy=False), desc=desc.astype(np.uint8, copy=False))
            chunk_paths.append(cp)
            for hist_id, rec, _ in computed:
                writer.writerow({
                    "hist_id": hist_id,
                    "raw_sha256": rec.sha256,
                    "phash64": rec.phash64,
                    "dhash64": rec.dhash64,
                    "width": rec.width,
                    "height": rec.height,
                })
            for count in counts.tolist():
                offsets.append(offsets[-1] + int(count))
            print(f"ORB/hash {min(start+len(subset), len(rows))}/{len(rows)}", flush=True)

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    np.save(out / "orb_offsets.npy", offsets_arr)
    total = int(offsets_arr[-1])
    xy_mm = np.lib.format.open_memmap(out / "orb_xy.npy", mode="w+", dtype=np.float32, shape=(total, 2))
    desc_mm = np.lib.format.open_memmap(out / "orb_desc.npy", mode="w+", dtype=np.uint8, shape=(total, 32))
    shapes_mm = np.lib.format.open_memmap(out / "orb_shapes.npy", mode="w+", dtype=np.int32, shape=(len(rows), 2))
    point_cursor = 0; row_cursor = 0
    for cp in chunk_paths:
        data = np.load(cp, allow_pickle=False)
        n_points = len(data["xy"]); n_rows = len(data["shapes"])
        xy_mm[point_cursor:point_cursor+n_points] = data["xy"]
        desc_mm[point_cursor:point_cursor+n_points] = data["desc"]
        shapes_mm[row_cursor:row_cursor+n_rows] = data["shapes"]
        point_cursor += n_points; row_cursor += n_rows
        del data
    xy_mm.flush(); desc_mm.flush(); shapes_mm.flush()
    del xy_mm, desc_mm, shapes_mm
    shutil.rmtree(temp)
    return manifest_path, total


def _encode_dino(rows, image_root: Path, model, out: Path, *, device: str, batch_size: int):
    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise TrackBError("CUDA requested for historical DINO indexing but unavailable")
    dev = torch.device(device)
    model = model.to(dev).eval()
    dest = np.lib.format.open_memmap(out / "dino_features.npy", mode="w+", dtype=np.float32, shape=(len(rows), FEATURE_WIDTH))
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start+batch_size]
            tensors = []
            for _hist_id, rel in subset:
                path = (image_root / rel).resolve()
                with Image.open(path) as im:
                    tensors.append(ctc_v2_eval_transform(im))
            x = torch.stack(tensors).to(dev, non_blocking=True)
            features = model.forward_features(x)
            features = model.forward_head(features, pre_logits=True)
            if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
                raise TrackBError(f"unexpected DINO audit feature shape: {tuple(features.shape)}")
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            dest[start:start+len(subset)] = features.cpu().numpy().astype(np.float32, copy=False)
            if start % (batch_size * 20) == 0:
                print(f"DINO {min(start+len(subset), len(rows))}/{len(rows)}", flush=True)
    dest.flush(); del dest

    mm = np.load(out / "dino_features.npy", mmap_mode="r")
    for start in range(0, len(mm), 4096):
        block = np.asarray(mm[start:start+4096], dtype=np.float32)
        if not np.isfinite(block).all():
            raise TrackBError("historical DINO feature package contains non-finite values")
        norms = np.linalg.norm(block, axis=1)
        if not np.all(np.abs(norms - 1.0) <= 1e-4):
            raise TrackBError("historical DINO features are not L2-normalized within tolerance")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the full 117,546-image Track-B historical comparison package.")
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--id-column", default="hist_id")
    ap.add_argument("--path-column", default="relative_path")
    ap.add_argument("--dino-checkpoint", required=True)
    ap.add_argument("--factory-manifest", required=True)
    ap.add_argument("--factory-source-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--execution-lock", required=True)
    ap.add_argument("--code-attestation", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--orb-chunk-size", type=int, default=512)
    ap.add_argument("--dino-batch-size", type=int, default=64)
    args = ap.parse_args()

    source_manifest = Path(args.source_manifest).resolve()
    image_root = Path(args.image_root).resolve()
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise TrackBError(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.dino_checkpoint) != DINO_AUDIT_SHA256:
        raise TrackBError("historical DINO checkpoint SHA mismatch")
    lock = load_json(args.execution_lock)
    validate_execution_lock(lock)
    if sha256_file(args.code_attestation) != str(lock.get("code_attestation_sha256", "")):
        raise TrackBError("historical builder code-attestation SHA differs from execution lock")
    verify_code_attestation(args.repo_root, args.code_attestation)
    rows = _read_source_manifest(source_manifest, id_col=args.id_column, path_col=args.path_column)
    started = time.perf_counter()
    model, dino_identity = _load_dino(args)
    hist_manifest, total_keypoints = _pack_orb(rows, image_root, out, workers=args.workers, chunk_size=args.orb_chunk_size)
    _encode_dino(rows, image_root, model, out, device=args.device, batch_size=args.dino_batch_size)

    files = {}
    for key, name in {
        "historical_manifest":"historical_manifest.csv",
        "dino_features":"dino_features.npy",
        "orb_offsets":"orb_offsets.npy",
        "orb_xy":"orb_xy.npy",
        "orb_desc":"orb_desc.npy",
        "orb_shapes":"orb_shapes.npy",
    }.items():
        p = out / name
        files[key] = {"path": name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    certificate = {
        "schema_version":"1.0",
        "status":"PASS",
        "kind":"trackb_historical_compare_build",
        "image_count":len(rows),
        "source_manifest_sha256":sha256_file(source_manifest),
        "dino_audit_encoder_sha256":DINO_AUDIT_SHA256,
        "code_attestation_sha256":sha256_file(args.code_attestation),
        "dino_identity":dino_identity,
        "audit_feature_preprocessing":{
            "entrypoint":"cropcop_je.data.ctc_v2_eval_transform",
            "ctc_v2_config_sha256":CTC_V2_CONFIG_SHA256,
            "source_commit":"604aafd51e20e70098ce4af647e90c8ff558a9e8",
        },
        "orb":{
            "max_side":800,
            "nfeatures":1200,
            "total_keypoints":total_keypoints,
        },
        "features_l2_normalized":True,
        "files":files,
        "built_at_utc":datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":time.perf_counter()-started,
    }
    atomic_write_json(out / "BUILD_CERTIFICATE.json", certificate)
    files["build_certificate"]={"path":"BUILD_CERTIFICATE.json","sha256":sha256_file(out / "BUILD_CERTIFICATE.json"),"bytes":(out / "BUILD_CERTIFICATE.json").stat().st_size}
    manifest = {
        "schema_version":"1.0",
        "role":"historical_compare",
        "image_count":EXPECTED_ROWS,
        "dino_audit_encoder_sha256":DINO_AUDIT_SHA256,
        "code_attestation_sha256":sha256_file(args.code_attestation),
        "features_l2_normalized":True,
        "ctc_v2_config_sha256":CTC_V2_CONFIG_SHA256,
        "files":files,
    }
    atomic_write_json(out / "TRACKB_INPUT_MANIFEST.json", manifest)
    print(json.dumps({"status":"PASS","output_dir":str(out),"elapsed_seconds":certificate["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
