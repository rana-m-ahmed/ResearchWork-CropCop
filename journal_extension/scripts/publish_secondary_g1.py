from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1_publication import ensure_private_target, wait_until_ready
from cropcop_je.hashing import sha256_file
from cropcop_je.secondary import validate_secondary_g1_bundle


def run(cmd: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed rc={cp.returncode}: {cmd[0]} {cmd[1] if len(cmd)>1 else ''}: {(cp.stderr or cp.stdout)[-1000:]}")
    return cp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--dataset-slug", required=True)
    ap.add_argument("--receipt", required=True)
    args = ap.parse_args()

    bundle = Path(args.bundle_dir).resolve()
    seal, errors = validate_secondary_g1_bundle(bundle)
    if errors:
        raise SystemExit("secondary G1 bundle invalid before publication: " + "; ".join(errors))
    slug = args.dataset_slug.strip()
    target = ensure_private_target(slug, env=os.environ)

    with tempfile.TemporaryDirectory() as td:
        package = Path(td) / "package"
        package.mkdir()
        for item in bundle.iterdir():
            dest = package / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        dataset_name = slug.split("/", 1)[1]
        (package / "dataset-metadata.json").write_text(json.dumps({
            "title": dataset_name,
            "id": slug,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        }, indent=2) + "\n", encoding="utf-8")
        run([
            "kaggle", "datasets", "version", "-p", str(package),
            "-m", f"CropCop secondary G1 {seal['secondary_g1_seal_sha256'][:16]}",
            "-q", "-r", "skip",
        ])
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        wait_until_ready(api, slug, timeout_seconds=900.0)

        roundtrip = Path(td) / "roundtrip"
        roundtrip.mkdir()
        run(["kaggle", "datasets", "download", "-d", slug, "-p", str(roundtrip), "--unzip", "-q"])
        rt_seal, rt_errors = validate_secondary_g1_bundle(roundtrip)
        if rt_errors:
            raise RuntimeError("round-trip secondary G1 validation failed: " + "; ".join(rt_errors))
        if rt_seal.get("secondary_g1_seal_sha256") != seal.get("secondary_g1_seal_sha256"):
            raise RuntimeError("round-trip secondary G1 seal SHA differs from local bundle")

    receipt = {
        "schema_version": "1.0",
        "status": "PASS",
        "dataset_slug": slug,
        "secondary_g1_seal_sha256": seal["secondary_g1_seal_sha256"],
        "source_git_sha": seal["source_git_sha"],
        "private_target_verified": target.get("status") == "PASS",
        "roundtrip_verified": True,
        "published_bundle_file_count": sum(1 for p in bundle.rglob("*") if p.is_file()),
        "seal_file_sha256": sha256_file(bundle / "SECONDARY_G1_MODEL_IDENTITY_SEAL.json"),
    }
    atomic_write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
