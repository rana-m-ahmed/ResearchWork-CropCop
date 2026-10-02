from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

TENSOR_IDENTITY_ALGORITHM = "cropcop-tensor-identity-v1"
_DOMAIN = b"CROPCOP:TENSOR-IDENTITY:v1\x00"


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "big", signed=False) + payload


def tensor_identity_sha256(state: Mapping[str, Any]) -> str:
    """Canonical CropCop tensor-state identity.

    Binds a versioned domain, sorted UTF-8 keys, dtype, shape, and raw
    contiguous CPU tensor bytes with explicit length framing.
    """
    h = hashlib.sha256()
    h.update(_DOMAIN)
    keys = sorted(str(key) for key in state)
    h.update(_frame(b"N", str(len(keys)).encode("ascii")))
    for key in keys:
        value = state[key].detach().cpu().contiguous()
        h.update(b"T")
        h.update(_frame(b"K", key.encode("utf-8")))
        h.update(_frame(b"D", str(value.dtype).encode("ascii")))
        shape = json.dumps([int(x) for x in value.shape], separators=(",", ":")).encode("ascii")
        h.update(_frame(b"S", shape))
        h.update(_frame(b"B", value.numpy().tobytes(order="C")))
    return h.hexdigest()
