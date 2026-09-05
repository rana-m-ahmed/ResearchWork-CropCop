from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.g1 import validate_g1_seal_object
from cropcop_je.hashing import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--dataset-slug", required=True, help="Pre-created private owner/dataset slug.")
    args = ap.parse_args()
    if "/" not in args.dataset_slug:
        raise SystemExit("dataset slug must be owner/dataset")
    bundle = Path(args.bundle_dir).resolve()
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    errors = validate_g1_seal_object(seal)
    if errors:
        raise SystemExit("refuse to publish invalid G1 bundle: " + "; ".join(errors))
    for key in ("S1", "S2", "S3"):
        row = seal["pair_initializations"][key]
        p = bundle / "private" / row["basename"]
        if not p.exists() or sha256_file(p) != row["sha256"]:
            raise SystemExit(f"G1 pair artifact missing/corrupt before private publication: {key}")

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "bundle"
        shutil.copytree(bundle, staging)
        metadata = {
            "title": args.dataset_slug.split("/", 1)[1],
            "id": args.dataset_slug,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        }
        (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(staging), "-m",
             f"CropCop sealed G1 {seal['g1_seal_sha256'][:12]}", "-q", "-r", "zip"],
            check=True, capture_output=True, text=True, timeout=1800,
        )
    print(json.dumps({"status": "PASS", "dataset_slug": args.dataset_slug,
                      "g1_seal_sha256": seal["g1_seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
