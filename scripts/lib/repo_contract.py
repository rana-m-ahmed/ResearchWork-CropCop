from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

EXPECTED = {
    "images": 109107,
    "classes": 120,
    "train": 76376,
    "val": 16368,
    "test": 16363,
    "pte_bytes": 23696352,
    "pte_sha256": "7c70d0f307f0d9578310600913cb8ff171b294ae4de0813f7cd1d6a53628bdf1",
}

FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pte", ".onnx", ".tflite", ".safetensors", ".npy", ".npz"}
FORBIDDEN_NAMES = {"private_evidence", "raw_images"}
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
