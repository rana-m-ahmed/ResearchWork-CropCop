from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json

PHASES = [
    "smoke-write",
    "smoke-restore",
    "dual-gpu-smoke",
    "g1",
    "calibration-dual",
    "principal-dual",
]
CHAIN = """smoke-write
→ fresh smoke-restore
→ independent audit
→ dual-gpu-smoke
→ independent audit
→ g1
→ calibration-dual
→ principal-dual"""
SUPERSEDED_ACTIVE_SOURCE = "67370145c9104edd52330b788c3b41b28f5cab87"


def _active_section(text: str) -> str:
    begin = "<!-- QA1_OPERATOR_BEGIN -->"
    end = "<!-- QA1_OPERATOR_END -->"
    if begin not in text or end not in text:
        return ""
    return text.split(begin, 1)[1].split(end, 1)[0]


def validate(repo_root: str | Path) -> dict:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    generator = (root / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
    match = re.search(r'^AUTHORIZED_SOURCE_SHA = "([0-9a-f]{40})"$', generator, re.MULTILINE)
    authorized = match.group(1) if match else ""
    if not authorized:
        errors.append("generator authorized source SHA missing/ambiguous")

    smoke_readme = (root / "journal_extension/kaggle/README_SMOKE.md").read_text(encoding="utf-8")
    example = json.loads((root / "journal_extension/kaggle/smoke_inputs.example.json").read_text(encoding="utf-8"))
    je_readme = (root / "journal_extension/README.md").read_text(encoding="utf-8")
    active = _active_section(je_readme)
    if not active:
        errors.append("journal_extension/README.md active QA1 operator section missing")

    if example.get("authorized_source_sha") != authorized:
        errors.append("smoke_inputs.example.json source differs from generator")
    if example.get("phase_vocabulary") != PHASES:
        errors.append("smoke_inputs.example.json phase vocabulary drift")

    for label, text in (("README_SMOKE.md", smoke_readme), ("journal_extension/README.md active section", active)):
        if authorized and authorized not in text:
            errors.append(f"{label} does not name generator-authorized source")
        if CHAIN not in text:
            errors.append(f"{label} chronology chain missing/drifted")
        for phase in PHASES:
            if phase not in text:
                errors.append(f"{label} missing phase {phase}")

    for label, text in (
        ("README_SMOKE.md", smoke_readme),
        ("smoke_inputs.example.json", json.dumps(example, sort_keys=True)),
        ("journal_extension/README.md active section", active),
    ):
        if SUPERSEDED_ACTIVE_SOURCE in text:
            errors.append(f"{label} names superseded active execution source")

    for token in (
        "CROPCOP_SMOKE_B_INPUT_ROOT",
        "CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT",
        "SMOKE_B_EVIDENCE.json",
        "DUAL_GPU_SMOKE_EVIDENCE.json",
        'EXECUTION_PHASE == "dual-gpu-smoke"',
        'EXECUTION_PHASE in {"g1", "calibration-dual", "principal-dual"}',
    ):
        if token not in generator:
            errors.append(f"generator missing operator-handoff token: {token}")

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "authorized_source_sha": authorized,
        "phases": PHASES,
        "chronology": CHAIN.splitlines(),
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--json-report", default="")
    args = ap.parse_args()
    report = validate(args.repo_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_report:
        atomic_write_json(args.json_report, report)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
