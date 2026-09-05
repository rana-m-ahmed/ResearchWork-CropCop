from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import factory_bundle_hash
from cropcop_je.hashing import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--source-root", default="", help="Root containing the complete trusted factory/helper source bundle; defaults to repo root.")
    ap.add_argument("--factory-spec", required=True, help="module:function")
    ap.add_argument("--file", action="append", dest="files", default=[],
                    help="Additional repository-relative helper source file affecting reconstruction.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if ":" not in args.factory_spec:
        raise SystemExit("factory spec must be module:function")
    repo = Path(args.repo_root).resolve()
    source_root = Path(args.source_root).resolve() if args.source_root else repo
    module_name, fn_name = args.factory_spec.split(":", 1)
    import sys
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    source = Path(inspect.getsourcefile(fn) or getattr(module, "__file__", "")).resolve()
    try:
        entry_rel = source.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise SystemExit("teacher factory entrypoint source must be inside the authorized repository tree") from exc

    paths = {entry_rel, *args.files}
    rows = []
    for rel in sorted(paths):
        p = (source_root / rel).resolve()
        try:
            p.relative_to(source_root)
        except ValueError as exc:
            raise SystemExit(f"factory source escapes repository: {rel}") from exc
        if not p.is_file():
            raise SystemExit(f"factory source file missing: {rel}")
        rows.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size})

    manifest = {
        "schema_version": "1.0",
        "entrypoint": args.factory_spec,
        "output_order_transform": "none",
        "files": rows,
    }
    manifest["bundle_sha256"] = factory_bundle_hash(manifest)
    atomic_write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
