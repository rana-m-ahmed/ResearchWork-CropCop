#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from lib.repo_contract import EXPECTED, FORBIDDEN_NAMES, FORBIDDEN_SUFFIXES, SECRET_PATTERNS

ROOT = Path(__file__).resolve().parents[1]

def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-report", default="")
    args = parser.parse_args()
    errors: list[str] = []
    registry = json.loads((ROOT / "metrics/metric_registry.json").read_text())
    results = list(csv.DictReader((ROOT / "metrics/main_results.csv").open()))
    claims = list(csv.DictReader((ROOT / "evidence/public/claim_evidence_matrix.csv").open()))
    for key in ("images", "classes", "train", "val", "test"):
        check(registry["dataset"][key] == EXPECTED[key], f"dataset.{key} mismatch", errors)
    check(registry["pte"]["bytes"] == EXPECTED["pte_bytes"], "PTE byte count mismatch", errors)
    check(registry["pte"]["sha256"] == EXPECTED["pte_sha256"], "PTE SHA-256 mismatch", errors)
    check({r["state"] for r in results} == {"reference", "mobile_float", "converted_int8", "pte_runtime"}, "model-state table incomplete", errors)
    blocked = {r["claim_id"] for r in claims if r["status"] == "blocked"}
    check({"C14", "C15", "C16", "C42"}.issubset(blocked), "blocked claim ledger is incomplete", errors)
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        check(not any(part in FORBIDDEN_NAMES for part in rel.parts), f"restricted path committed: {rel}", errors)
        check(path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"restricted binary committed: {rel}", errors)
        if path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f"file exceeds 25 MiB public threshold: {rel}")
        if path.suffix.lower() in {".md", ".tex", ".bib", ".json", ".csv", ".py", ".sh", ".yml", ".yaml", ".cff", ".txt"} or path.name == "Makefile":
            text = path.read_text(encoding="utf-8", errors="replace")
            if rel != Path("scripts/validate_repository.py"):
                check("/mnt/data/" not in text and "/home/oai/" not in text, f"private absolute path in {rel}", errors)
            for pattern in SECRET_PATTERNS:
                check(pattern.search(text) is None, f"possible secret in {rel}", errors)
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": []}
    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
