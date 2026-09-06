from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import MNV4_MODEL_NAME, TIMM_VERSION
from cropcop_je.hashing import sha256_file, sha256_json


def tensor_identity_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8") + b"\0")
        h.update(str(value.dtype).encode("ascii") + b"\0")
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        h.update(value.numpy().tobytes(order="C"))
    return h.hexdigest()


def _cfg_dict(model) -> dict:
    cfg = getattr(model, "pretrained_cfg", None)
    if cfg is None:
        raise RuntimeError("timm model exposes no pretrained_cfg")
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    import timm
    if timm.__version__ != TIMM_VERSION:
        raise SystemExit(f"timm drift: expected {TIMM_VERSION}, got {timm.__version__}")

    model = timm.create_model(MNV4_MODEL_NAME, pretrained=True, num_classes=1000)
    cfg = _cfg_dict(model)
    hf_id = cfg.get("hf_hub_id")
    url = cfg.get("url")
    if hf_id:
        source_kind = "timm_pretrained_cfg_hf_hub"
        source_locator = str(hf_id)
    elif url:
        source_kind = "timm_pretrained_cfg_url"
        source_locator = str(url)
    else:
        raise SystemExit("official timm pretrained_cfg exposes neither hf_hub_id nor URL")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / "MNV4_PRETRAINED.safetensors"
    if candidate.exists():
        raise SystemExit(f"refuse to overwrite existing MNV4 preparation candidate: {candidate}")

    from safetensors.torch import save_file
    state = OrderedDict(
        (key, value.detach().cpu().contiguous())
        for key, value in sorted(model.state_dict().items())
    )
    save_file(state, str(candidate))

    record = {
        "schema_version": "1.0",
        "status": "PASS",
        "model_name": MNV4_MODEL_NAME,
        "timm_version": TIMM_VERSION,
        "upstream_source_kind": source_kind,
        "upstream_source_locator": source_locator,
        "upstream_pretrained_cfg_sha256": sha256_json(cfg),
        "tensor_identity_sha256": tensor_identity_sha256(state),
        "candidate_path": str(candidate),
        "candidate_basename": candidate.name,
        "candidate_serialization_format": "safetensors_state_dict",
        "candidate_sha256": sha256_file(candidate),
        "candidate_bytes": candidate.stat().st_size,
        "scientific_result_produced": False,
    }
    atomic_write_json(args.record, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
