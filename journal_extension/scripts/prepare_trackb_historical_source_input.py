from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import TrackBError

EXPECTED_ROWS = 117546


def main() -> int:
    ap=argparse.ArgumentParser(description="Prepare the raw V4 117,546-image source package for the Track-B historical-index builder.")
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--source-manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--id-column", default="hist_id")
    ap.add_argument("--path-column", default="relative_path")
    args=ap.parse_args()
    root=Path(args.package_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    manifest=(root/args.source_manifest).resolve(); image_root=(root/args.image_root).resolve()
    for p in (manifest,image_root):
        if root not in p.parents and p != root: raise TrackBError(f"path escapes package root: {p}")
    if not manifest.is_file() or not image_root.is_dir(): raise TrackBError("source manifest/image root missing")
    count=0; ids=set(); rels=set()
    with manifest.open(encoding="utf-8",newline="") as fh:
        reader=csv.DictReader(fh)
        missing={args.id_column,args.path_column}.difference(reader.fieldnames or [])
        if missing: raise TrackBError(f"historical source manifest missing columns: {sorted(missing)}")
        for row in reader:
            hist_id=str(row[args.id_column]).strip(); rel=str(row[args.path_column]).strip().replace('\\','/')
            posix=PurePosixPath(rel)
            if not hist_id or hist_id in ids: raise TrackBError(f"missing/duplicate historical ID: {hist_id!r}")
            if not rel or posix.is_absolute() or '..' in posix.parts or rel in rels: raise TrackBError(f"unsafe/duplicate historical path: {rel!r}")
            ids.add(hist_id); rels.add(rel); count+=1
    if count != EXPECTED_ROWS: raise TrackBError(f"expected {EXPECTED_ROWS} historical rows, got {count}")
    out={
        "schema_version":"1.0", "role":"historical_source", "row_count":count,
        "image_root":image_root.relative_to(root).as_posix(), "id_column":args.id_column, "path_column":args.path_column,
        "files":{"source_manifest":{"path":manifest.relative_to(root).as_posix(),"sha256":sha256_file(manifest),"bytes":manifest.stat().st_size}},
    }
    atomic_write_json(root/'TRACKB_HIST_SOURCE_MANIFEST.json',out)
    print(json.dumps({"status":"PASS","role":"historical_source","manifest":str(root/'TRACKB_HIST_SOURCE_MANIFEST.json')},indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
