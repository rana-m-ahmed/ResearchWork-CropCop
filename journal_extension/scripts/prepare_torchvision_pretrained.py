from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.secondary import BASELINE_SPECS, TORCHVISION_VERSION, official_torchvision_identity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=sorted(BASELINE_SPECS), required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    import torchvision
    from torch.hub import download_url_to_file

    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise SystemExit(
            f"torchvision drift: expected {TORCHVISION_VERSION}, got {torchvision.__version__}"
        )

    _builder, weights, spec = official_torchvision_identity(args.model_key)
    source_url = str(weights.url)
    if Path(urlparse(source_url).path).name != spec["official_filename"]:
        raise SystemExit("TorchVision weight URL filename differs from frozen identity")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / spec["official_filename"]
    if artifact.exists():
        raise SystemExit(f"refuse to overwrite existing pretrained artifact: {artifact}")

    download_url_to_file(
        source_url,
        str(artifact),
        hash_prefix=spec["official_sha256_prefix"],
        progress=True,
    )
    digest = sha256_file(artifact)
    if not digest.startswith(spec["official_sha256_prefix"]):
        raise SystemExit("downloaded TorchVision artifact failed official hash-prefix verification")

    record = {
        "schema_version": "1.0",
        "status": "PASS",
        "model_key": args.model_key,
        "model_name": spec["model_name"],
        "torchvision_version": TORCHVISION_VERSION,
        "weight_enum": spec["weight_enum"],
        "official_filename": spec["official_filename"],
        "source_kind": "torchvision_weight_enum_url",
        "source_locator": source_url,
        "artifact_path": str(artifact),
        "artifact_sha256": digest,
        "artifact_bytes": artifact.stat().st_size,
        "scientific_result_produced": False,
    }
    atomic_write_json(args.record, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
