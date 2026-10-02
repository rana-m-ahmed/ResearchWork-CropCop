from __future__ import annotations

import json
from pathlib import Path


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook():
    cells = [
        md("""# CropCop Track B — Core Input Package Builder

**Input-preparation notebook only; no protected external inference.**

Attach these exact Kaggle datasets:

- Final V1: ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1
- R07 S1: sabahatabbas/sec-je-r07-cnxtt-context-s1-8904b100d223-a01
- R07 S2: sabahatabbas/cropcop-r07-cnxtt-context-s2-abce1197-56023042
- R07 S3: sabahatabbas/cropcop-r07-cnxtt-context-s3-f13ca687-56023042
- DINO audit bundle: ranamuhammadahmed6/cropcop-secondary-g1-8904b100, **version 2**

Enable Internet only for cloning the Track-B repository branch. No GPU is required. The builder discovers checkpoint/run-record/factory files by authoritative SHA-256, copies **only** the 16,368-image V1 validation surface, and refuses ambiguous identities.
"""),
        md("""## Output contract

A successful run creates /kaggle/working/trackb_core_v1/ with TRACKB_INPUT_MANIFEST.json (role = core) and CORE_BUILD_CERTIFICATE.json (status = PASS).

Publish that output directory as an immutable private Kaggle Dataset. The final Track-B claim notebook consumes the published core package; it does not consume these source datasets directly.
"""),
        code("""from pathlib import Path
import json, shutil, subprocess, sys

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
REPO = WORK / 'ResearchWork-CropCop-trackb'
OUTPUT = WORK / 'trackb_core_v1'
REPO_URL = 'https://github.com/rana-m-ahmed/ResearchWork-CropCop.git'
BRANCH = 'trackb-r07-infrastructure-20260919'

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    ['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(REPO)],
    check=True,
)
print('Repository head:', subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip())
"""),
        code("""SLUGS = {
    'final_v1': 'cropcop-finalized-v8-11-2026-1',
    'r07_s1': 'sec-je-r07-cnxtt-context-s1-8904b100d223-a01',
    'r07_s2': 'cropcop-r07-cnxtt-context-s2-abce1197-56023042',
    'r07_s3': 'cropcop-r07-cnxtt-context-s3-f13ca687-56023042',
    'dino': 'cropcop-secondary-g1-8904b100',
}

def resolve_mount(slug_basename):
    direct = INPUT / slug_basename
    candidates = []
    if direct.is_dir():
        candidates.append(direct.resolve())
    candidates.extend(
        p.resolve() for p in INPUT.rglob(slug_basename)
        if p.is_dir() and p.resolve() not in candidates
    )
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one mounted dataset named {slug_basename!r}; found {candidates}')
    return candidates[0]

mounts = {key: resolve_mount(value) for key, value in SLUGS.items()}
print(json.dumps({k: str(v) for k, v in mounts.items()}, indent=2))
"""),
        code("""builder = REPO / 'journal_extension/scripts/build_trackb_core_package.py'
if not builder.is_file():
    raise RuntimeError(f'Core builder missing: {builder}')
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise RuntimeError(f'Output must be empty: {OUTPUT}')

cmd = [
    sys.executable, str(builder),
    '--repo-root', str(REPO),
    '--final-v1-root', str(mounts['final_v1']),
    '--r07-s1-root', str(mounts['r07_s1']),
    '--r07-s2-root', str(mounts['r07_s2']),
    '--r07-s3-root', str(mounts['r07_s3']),
    '--dino-bundle-root', str(mounts['dino']),
    '--output-root', str(OUTPUT),
]
subprocess.run(cmd, cwd=REPO, check=True)
"""),
        code("""manifest = json.loads((OUTPUT / 'TRACKB_INPUT_MANIFEST.json').read_text())
certificate = json.loads((OUTPUT / 'CORE_BUILD_CERTIFICATE.json').read_text())

if manifest.get('role') != 'core':
    raise RuntimeError('Core package role mismatch')
if certificate.get('status') != 'PASS':
    raise RuntimeError('Core build certificate is not PASS')
if certificate.get('validation_rows') != 16368:
    raise RuntimeError('Core package validation row count mismatch')
if certificate.get('consumed_test_image_bytes_copied') is not False:
    raise RuntimeError('Consumed-test safety assertion failed')
if (OUTPUT / 'v1_validation' / 'test').exists():
    raise RuntimeError('Forbidden V1-test directory exists in core output')

print(json.dumps(certificate, indent=2, sort_keys=True))
print('PASS — publish /kaggle/working/trackb_core_v1 as an immutable private Kaggle Dataset.')
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    out = Path(__file__).resolve().parents[1] / "kaggle" / "trackb_build_core_package.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
