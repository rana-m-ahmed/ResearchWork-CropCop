from __future__ import annotations

import json
from pathlib import Path


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook() -> dict:
    cells = [
        md("""# CropCop Track B — R07 Internal Preflight Surface

**Not a claim-run notebook.** The only supported protected Track-B execution surface is `trackb_r07_master.ipynb`.

This notebook exists for inspection and frozen-input/environment preflight only. It intentionally cannot start external R07 inference because protected execution requires the master's durable attempt ledger, source acquisition, scratch isolation, private evidence round-trip, and terminal publication controls.
"""),
        code("""from pathlib import Path
import json, shutil, subprocess, sys

INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working/trackb_r07_preflight')
SCRATCH_ROOT = Path('/kaggle/tmp/cropcop_trackb_r07_preflight')

manifests = []
for p in sorted(INPUT_ROOT.glob('**/TRACKB_INPUT_MANIFEST.json')):
    obj = json.loads(p.read_text(encoding='utf-8'))
    manifests.append((obj.get('role'), p, obj))
roles = [r for r, _, _ in manifests]
required = {'core', 'historical_compare', 'gvlid_v5', 'irish_potato'}
if set(roles) != required:
    raise RuntimeError(f'Expected exactly {sorted(required)}, got {sorted(roles)}')

core_path, core = next((p, o) for r, p, o in manifests if r == 'core')
repo_rel = str(core.get('repository_root', '')).strip()
if not repo_rel:
    raise RuntimeError('core TRACKB_INPUT_MANIFEST.json must define repository_root')
REPO_ROOT = (core_path.parent / repo_rel).resolve()
RUNNER = REPO_ROOT / 'journal_extension/scripts/run_trackb_r07.py'
if not RUNNER.is_file():
    raise RuntimeError(f'Track-B runner missing: {RUNNER}')

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
if SCRATCH_ROOT.exists():
    shutil.rmtree(SCRATCH_ROOT)

print('Track-B input roles:', roles)
print('Repository root:', REPO_ROOT)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], check=True)
"""),
        code("""cmd = [
    sys.executable, str(RUNNER),
    '--input-root', str(INPUT_ROOT),
    '--output-root', str(OUTPUT_ROOT),
    '--scratch-root', str(SCRATCH_ROOT),
    '--device', 'cuda:0',
    '--workers', '4',
    '--mode', 'preflight',
]
print('Launching prediction-free preflight:', ' '.join(cmd))
subprocess.run(cmd, cwd=REPO_ROOT, check=True)

gate = json.loads((OUTPUT_ROOT / 'PREFLIGHT_PASS.json').read_text(encoding='utf-8'))
if gate.get('status') != 'PASS':
    raise RuntimeError('Track-B preflight did not PASS')
print(json.dumps(gate, indent=2, sort_keys=True))
"""),
        md("""## Boundary

A PASS here proves only the frozen authority/environment/R07-family replay and input-package preflight. It does **not** authorize manuscript claims or protected external inference. Use the master notebook for the controlled claim run.
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
    out = Path(__file__).resolve().parents[1] / 'kaggle' / 'trackb_r07_end_to_end.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
