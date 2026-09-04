from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

EXPECTED = {
    "audited_images": 117546,
    "v5_images": 109150,
    "images": 109107,
    "classes": 120,
    "train": 76376,
    "val": 16368,
    "test": 16363,
    "historical_v4_edges": 8672,
    "corrected_v5_edges": 8573,
    "exact_edges": 6196,
    "strong_hash_edges": 445,
    "feature_edges_reopened": 2031,
    "feature_edges_retained": 1932,
    "feature_edges_rejected": 99,
    "cross_label_subset": 17,
    "duplicate_removals": 8355,
    "manual_review_deletions": 43,
    "source_family_rows": 109107,
    "exact_source_path_rows": 84146,
    "source_family_only_rows": 24961,
    "registered_sources": 15,
    "final_source_families": 10,
    "final_manifest_fingerprint": "7c368e6e3d8be3bb3a9a3a5f961075d4faa125bcac2e98a3b55e1a1c61f1c523",
    "final_build_fingerprint": "2d7c237981b8943d9b08a522db0489849a4bfe7f488dfb6461489dd36dcdc12b",
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
