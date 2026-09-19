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
        md("""# CropCop Track B — R07 Automated Master

This is the **single operator-facing Track-B notebook**.

One-time Kaggle setup:
- add secret KAGGLE_API_TOKEN;
- add secret CROPCOP_GITHUB_TOKEN;
- enable Internet;
- select **T4 x2**.

Do **not** manually attach or publish Track-B input/evidence datasets.

The notebook creates a clean isolated Track-B Python environment under /kaggle/working, installs the frozen scientific stack there (including the official PyTorch 2.12.1 / torchvision 0.27.1 CUDA 12.6 wheels), then runs the automated Track-B controller from that environment. Kaggle's preinstalled Torch stack is deliberately left untouched.

The consumed V1 test remains forbidden. The classifier family/seeds are frozen. GVLiD is the confirmatory field cohort; Irish Potato is the complementary harder stress cohort. Candidate substitution after any external prediction is forbidden.
"""),
        code("""from pathlib import Path
import json, os, shutil, subprocess, sys

WORK = Path('/kaggle/working')
REPO = WORK / 'ResearchWork-CropCop-trackb-v2'
REPO_URL = 'https://github.com/rana-m-ahmed/ResearchWork-CropCop.git'
BRANCH = 'trackb-r07-infrastructure-20260919'

if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(
    ['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, str(REPO)],
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
"""),
        code("""BOOTSTRAP = REPO / 'journal_extension/scripts/bootstrap_trackb_runtime.py'
VENV_ROOT = WORK / 'trackb_runtime_env'
BOOTSTRAP_RECEIPT = WORK / 'TRACKB_RUNTIME_BOOTSTRAP.json'

if not BOOTSTRAP.is_file():
    raise RuntimeError(f'Track-B runtime bootstrap missing: {BOOTSTRAP}')

subprocess.run(
    [
        sys.executable,
        str(BOOTSTRAP),
        '--venv-root', str(VENV_ROOT),
        '--receipt', str(BOOTSTRAP_RECEIPT),
    ],
    cwd=REPO,
    check=True,
    env=os.environ.copy(),
)

bootstrap = json.loads(BOOTSTRAP_RECEIPT.read_text())
if bootstrap.get('status') != 'PASS':
    raise RuntimeError('Track-B isolated runtime bootstrap did not PASS')
VENV_PY = Path(bootstrap['venv_python'])
if not VENV_PY.is_file():
    raise RuntimeError(f'Isolated Track-B Python missing: {VENV_PY}')
print(json.dumps({
    'runtime_status': bootstrap['status'],
    'runtime_mode': bootstrap['mode'],
    'python': bootstrap['probe']['python'],
    'versions': bootstrap['probe']['versions'],
    'torch_cuda_version': bootstrap['probe']['torch_cuda_version'],
    'cuda_available': bootstrap['probe']['cuda_available'],
    'cuda_devices': bootstrap['probe']['cuda_devices'],
}, indent=2, sort_keys=True))
"""),
        code("""MASTER = REPO / 'journal_extension/scripts/run_trackb_r07_master.py'
if not MASTER.is_file():
    raise RuntimeError(f'Automated Track-B controller missing: {MASTER}')

WORKSPACE = WORK / 'trackb_master'
if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE)

subprocess.run(
    [
        str(VENV_PY),
        str(MASTER),
        '--repo-root', str(REPO),
        '--workspace', str(WORKSPACE),
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

if receipt.get('status') != 'PASS_AUTOMATED_TRACK_B_COMPLETE':
    raise RuntimeError('Automation receipt is not terminal PASS')
if closure.get('status') != 'TRACK_B_CLOSED' or qa.get('status') != 'PASS':
    raise RuntimeError('Track-B terminal scientific closure/QA is not PASS')

print(json.dumps({
    'automation_status': receipt['status'],
    'track_b_status': closure['status'],
    'closure_sha256': closure['closure_sha256'],
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
