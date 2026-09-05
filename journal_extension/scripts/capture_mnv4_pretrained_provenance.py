from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import MNV4_MODEL_NAME, TIMM_VERSION
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.models import _state_dict_from_file


def _cfg_dict(model) -> dict:
    cfg = getattr(model, "pretrained_cfg", None)
    if cfg is None:
        raise RuntimeError("timm model exposes no pretrained_cfg")
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="Exact candidate pretrained bytes to independently verify.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import timm
    import torch

    if timm.__version__ != TIMM_VERSION:
        raise SystemExit(f"timm drift: expected {TIMM_VERSION}, got {timm.__version__}")

    candidate_path = Path(args.artifact)
    candidate = _state_dict_from_file(candidate_path)

    # This call resolves the official pretrained object through timm's own locked pretrained_cfg.
    official = timm.create_model(MNV4_MODEL_NAME, pretrained=True, num_classes=1000)
    official_state = official.state_dict()
    if set(candidate) != set(official_state):
        raise SystemExit("candidate state_dict keys differ from the official timm pretrained object")
    mismatches = []
    for key in official_state:
        a = candidate[key]
        b = official_state[key].detach().cpu()
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype or not torch.equal(a.detach().cpu(), b):
            mismatches.append(key)
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise SystemExit(f"candidate tensors differ from official timm pretrained object: {mismatches}")

    cfg = _cfg_dict(official)
    hf_id = cfg.get("hf_hub_id")
    url = cfg.get("url")
    if hf_id:
        source_kind = "timm_pretrained_cfg_hf_hub"
        source_locator = str(hf_id)
    elif url:
        source_kind = "timm_pretrained_cfg_url"
        source_locator = str(url)
    else:
        raise SystemExit("timm pretrained_cfg exposes neither hf_hub_id nor URL; official provenance cannot be sealed")

    record = {
        "schema_version": "1.0",
        "model_name": MNV4_MODEL_NAME,
        "timm_version": TIMM_VERSION,
        "source_kind": source_kind,
        "source_locator": source_locator,
        "timm_pretrained_cfg_sha256": sha256_json(cfg),
        "timm_pretrained_cfg": cfg,
        "artifact_sha256": sha256_file(candidate_path),
        "artifact_bytes": candidate_path.stat().st_size,
        "artifact_basename": candidate_path.name,
        "verification": "tensor_exact_match_against_timm_pretrained",
        "official_timm_tensor_match": True,
    }
    atomic_write_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
