from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.models import _state_dict_from_file
from cropcop_je.secondary import (
    BASELINE_SPECS,
    TORCHVISION_VERSION,
    official_torchvision_identity,
    validate_torchvision_provenance,
)
from cropcop_je.tensor_identity import TENSOR_IDENTITY_ALGORITHM, tensor_identity_sha256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=sorted(BASELINE_SPECS), required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    import torchvision

    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise SystemExit(
            f"torchvision drift: expected {TORCHVISION_VERSION}, got {torchvision.__version__}"
        )

    builder, weights, spec = official_torchvision_identity(args.model_key)
    artifact = Path(args.artifact)
    if artifact.name != spec["official_filename"]:
        raise SystemExit("candidate TorchVision filename differs from frozen official filename")
    digest = sha256_file(artifact)
    if not digest.startswith(spec["official_sha256_prefix"]):
        raise SystemExit("candidate TorchVision artifact failed official hash-prefix verification")

    candidate = _state_dict_from_file(artifact)
    official_model = builder(weights=weights)
    official_state = official_model.state_dict()
    if set(candidate) != set(official_state):
        raise SystemExit("candidate state keys differ from official TorchVision weight state")
    mismatches = []
    for key in official_state:
        a = candidate[key].detach().cpu()
        b = official_state[key].detach().cpu()
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype or not torch.equal(a, b):
            mismatches.append(key)
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise SystemExit(f"candidate tensors differ from official TorchVision state: {mismatches}")

    source_url = str(weights.url)
    if Path(urlparse(source_url).path).name != spec["official_filename"]:
        raise SystemExit("official TorchVision URL filename differs from frozen identity")

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
        "artifact_sha256": digest,
        "artifact_bytes": artifact.stat().st_size,
        "tensor_identity_algorithm": TENSOR_IDENTITY_ALGORITHM,
        "tensor_identity_sha256": tensor_identity_sha256(candidate),
        "official_tensor_identity_algorithm": TENSOR_IDENTITY_ALGORITHM,
        "official_tensor_identity_sha256": tensor_identity_sha256(official_state),
        "official_tensor_match": True,
        "verification": "exact_tensor_match_against_frozen_torchvision_weight_enum",
        "scientific_result_produced": False,
    }
    errors = validate_torchvision_provenance(record, model_key=args.model_key, artifact_path=artifact)
    if errors:
        raise SystemExit("TorchVision provenance validation failed: " + "; ".join(errors))
    atomic_write_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
