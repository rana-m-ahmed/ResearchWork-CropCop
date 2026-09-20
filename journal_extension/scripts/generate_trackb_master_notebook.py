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
        md("""# CropCop Track B — R07 Automated Master v3

This is the **single operator-facing Track-B notebook**.

One-time Kaggle setup:
- add secret KAGGLE_API_TOKEN;
- add secret CROPCOP_GITHUB_TOKEN;
- enable Internet;
- select **T4 x2**.

Do **not** manually attach or publish Track-B input/evidence datasets.

Kaggle's stock image is allowed to start with a different Torch stack. Before any scientific import, the notebook applies the exact Track-B requirements lock to the active Kaggle interpreter using the same execution pattern already qualified by CropCop Track A. Scientific code is then launched only in a **fresh subprocess**, so stale modules from the notebook process cannot contaminate Track B.

The consumed V1 test remains forbidden. The classifier family/seeds are frozen. GVLiD is the frozen external grape cohort (documented source acquisition includes in-situ vineyard and ex-situ imagery); Irish Potato is the complementary harder stress cohort. Candidate substitution after any external prediction is forbidden. Large mutable inputs use /kaggle/tmp; only final evidence is retained under /kaggle/working.
"""),
        code("""from pathlib import Path
import json, os, shutil, subprocess, sys

WORK = Path('/kaggle/working')
REPO = WORK / 'ResearchWork-CropCop-trackb-v3'
REPO_URL = 'https://github.com/rana-m-ahmed/ResearchWork-CropCop.git'
SOURCE_REF = 'trackb-r07-remediation-v3-20260920'

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    ['git', 'clone', '--depth', '1', '--branch', SOURCE_REF, REPO_URL, str(REPO)],
    check=True,
)
HEAD = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
print('Track-B source head:', HEAD)

from kaggle_secrets import UserSecretsClient
_secret_client = UserSecretsClient()
for _name in ('KAGGLE_API_TOKEN', 'CROPCOP_GITHUB_TOKEN'):
    _value = (_secret_client.get_secret(_name) or '').strip()
    if not _value:
        raise RuntimeError(f'Required Kaggle secret is empty: {_name}')
    if any(ch.isspace() for ch in _value):
        raise RuntimeError(f'Required Kaggle secret contains whitespace/newline: {_name}')
    os.environ[_name] = _value
del _value, _secret_client
print('Required secrets loaded into process environment: PASS')

# Fail fast on the exact Git transport used by terminal evidence publication.
# This imports no scientific package and runs before the expensive runtime repair.
sys.path.insert(0, str(REPO / 'journal_extension' / 'src'))
from cropcop_je.trackb_r07_ops import verify_github_repository_push_access
github_preflight = verify_github_repository_push_access(
    REPO,
    source_git_sha=HEAD,
)
print('GitHub evidence write preflight: PASS')
print(json.dumps(github_preflight, indent=2, sort_keys=True))
"""),
        code("""BOOTSTRAP = REPO / 'journal_extension/scripts/bootstrap_trackb_runtime.py'
LOCKFILE = REPO / 'journal_extension/track_b_r07/requirements-trackb.lock.txt'
BOOTSTRAP_RECEIPT = WORK / 'TRACKB_RUNTIME_BOOTSTRAP.json'

if not BOOTSTRAP.is_file():
    raise RuntimeError(f'Track-B runtime bootstrap missing: {BOOTSTRAP}')
if not LOCKFILE.is_file():
    raise RuntimeError(f'Track-B requirements lock missing: {LOCKFILE}')

subprocess.run(
    [
        sys.executable,
        str(BOOTSTRAP),
        '--requirements', str(LOCKFILE),
        '--receipt', str(BOOTSTRAP_RECEIPT),
    ],
    cwd=REPO,
    check=True,
    env=os.environ.copy(),
)

bootstrap = json.loads(BOOTSTRAP_RECEIPT.read_text())
if bootstrap.get('status') != 'PASS':
    raise RuntimeError('Track-B runtime repair/verification did not PASS')
if bootstrap.get('scientific_execution_requires_fresh_subprocess') is not True:
    raise RuntimeError('Track-B runtime receipt does not require a fresh scientific subprocess')

print(json.dumps({
    'runtime_status': bootstrap['status'],
    'runtime_mode': bootstrap['mode'],
    'python_expected': bootstrap['python_expected'],
    'pre_install_drift': bootstrap['pre_install_drift'],
    'versions': bootstrap['probe']['versions'],
    'torch_runtime_version': bootstrap['probe']['torch_runtime_version'],
    'torch_cuda_version': bootstrap['probe']['torch_cuda_version'],
    'cuda_available': bootstrap['probe']['cuda_available'],
    'cuda_devices': bootstrap['probe']['cuda_devices'],
    'opencv_runtime_version': bootstrap['probe']['opencv_runtime_version'],
}, indent=2, sort_keys=True))
"""),
        code("""MASTER = REPO / 'journal_extension/scripts/run_trackb_r07_master.py'
if not MASTER.is_file():
    raise RuntimeError(f'Automated Track-B controller missing: {MASTER}')

WORKSPACE = WORK / 'trackb_master'
if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE)

# IMPORTANT: scientific execution starts in a fresh interpreter after the exact
# package lock has been repaired. Do not import torch/torchvision/timm in the
# notebook process before this point.
subprocess.run(
    [
        sys.executable,
        str(MASTER),
        '--repo-root', str(REPO),
        '--workspace', str(WORKSPACE),
        '--scratch-root', '/kaggle/tmp/cropcop_trackb_r07',
        '--device', 'cuda:0',
    ],
    cwd=REPO,
    check=True,
    env=os.environ.copy(),
)
"""),
        code("""OUT = Path('/kaggle/working/trackb_master/trackb_r07')
receipt = json.loads((OUT / 'TRACKB_AUTOMATION_RECEIPT.json').read_text())
closure = json.loads((OUT / 'TRACKB_FINAL_CLOSURE.json').read_text())
qa = json.loads((OUT / 'TRACKB_FINAL_QA.json').read_text())

if closure.get('status') != 'TRACK_B_CLOSED' or qa.get('status') != 'PASS':
    raise RuntimeError('Track-B terminal scientific closure/QA is not PASS')
if receipt.get('status') != 'PASS_AUTOMATED_TRACK_B_COMPLETE':
    if receipt.get('status') == 'TRACK_B_CLOSED_PRIVATE_EVIDENCE_ARCHIVED_GITHUB_PUBLICATION_FAILED':
        raise RuntimeError(
            'Track B science is CLOSED and restricted evidence is safely archived, '
            'but GitHub public-safe publication failed after retries. Do not rerun '
            'scientific execution; repair GitHub publication using the preserved evidence.'
        )
    raise RuntimeError(f"Automation receipt is not terminal PASS: {receipt.get('status')}")

print(json.dumps({
    'automation_status': receipt['status'],
    'track_b_status': closure['status'],
    'closure_sha256': closure['closure_sha256'],
    'trackb_science_sha256': closure['trackb_science_sha256'],
    'final_qa_sha256': qa['qa_sha256'],
    'github_public_evidence': receipt['github_public_evidence'],
    'private_kaggle_evidence': receipt['private_kaggle_evidence'],
}, indent=2, sort_keys=True))
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
    out = Path(__file__).resolve().parents[1] / "kaggle" / "trackb_r07_master.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
