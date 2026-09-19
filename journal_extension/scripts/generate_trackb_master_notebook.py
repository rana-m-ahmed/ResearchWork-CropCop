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

Do **not** manually attach or publish Track-B input/evidence datasets. The controller acquires the frozen source assets, builds/reuses the private historical cache, acquires GVLiD v5 and Irish Potato Version 01 from their authoritative public repositories, runs the prediction-blind audits and protected R07 evaluation, archives restricted evidence to a private Kaggle Dataset, and publishes only audited public-safe evidence to GitHub.

The consumed V1 test remains forbidden. The classifier family/seeds are frozen. GVLiD is the confirmatory field cohort; Irish Potato is the complementary harder stress cohort. Candidate substitution after any external prediction is forbidden.
"""),
        code("""from pathlib import Path
import importlib.metadata as metadata
import shutil, subprocess, sys

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

expected = {
    'torch': '2.12.1',
    'torchvision': '0.27.1',
    'numpy': '2.5.2',
    'Pillow': '12.3.0',
    'opencv-python-headless': '4.12.0.88',
}
observed = {name: metadata.version(name) for name in expected}
print('Scientific runtime:', observed)
if observed != expected:
    raise RuntimeError(
        'Scientific dependency mismatch. This notebook intentionally fails closed rather than '
        'hot-swapping torch/torchvision inside an active Kaggle kernel. '
        f'Expected {expected}, observed {observed}.'
    )
"""),
        code("""MASTER = REPO / 'journal_extension/scripts/run_trackb_r07_master.py'
if not MASTER.is_file():
    raise RuntimeError(f'Automated Track-B controller missing: {MASTER}')

subprocess.run(
    [
        sys.executable,
        str(MASTER),
        '--repo-root', str(REPO),
        '--workspace', '/kaggle/working/trackb_master',
        '--device', 'cuda:0',
    ],
    cwd=REPO,
    check=True,
)
"""),
        code("""import json
from pathlib import Path

OUT = Path('/kaggle/working/trackb_master/trackb_r07')
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
