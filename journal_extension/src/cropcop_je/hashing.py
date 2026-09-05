from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any

def sha256_file(path:str|Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def canonical_json_bytes(obj:Any)->bytes:
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
def sha256_json(obj:Any)->str: return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()
def require_sha256(path:str|Path,expected:str,label:str)->str:
    actual=sha256_file(path)
    if actual.lower()!=expected.lower(): raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual
