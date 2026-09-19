from __future__ import annotations

import json
from pathlib import Path


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook() -> dict:
    cells = [
        md("""# CropCop Track B — R07 External Validation\n\n**Internal claim-engine notebook.** The supported operator entry point is `trackb_r07_master.ipynb`; it prepares these four roles automatically. This notebook remains as an inspectable/recovery execution surface and should not be used as a manual data-upload workflow. The scientific runner uses `cuda:0`; the second T4 is not a scientific dependency.\n\nScientific order: authority/replay → source verification → prediction-blind family/overlap audit → candidate seals → prediction firewall → R07 S1/S2/S3 inference → fixed family bootstrap → independent QA → `TRACK_B_CLOSED`.\n\nThe consumed V1 test is forbidden. No training, external tuning, candidate shopping, seed replacement, mapped-logit renormalization, or post-result remapping is allowed."""),
        md("""## Internal execution contract\n\nThe master controller verifies the exact scientific dependency stack and constructs the required `core`, `historical_compare`, `gvlid_v5`, and `irish_potato` roles before invoking the runner. This notebook does not acquire data, publish evidence, or hold credentials."""),
        code("""from pathlib import Path\nimport importlib.metadata as metadata\nimport json, os, shutil, subprocess, sys\n\nINPUT_ROOT = Path('/kaggle/input')\nOUTPUT_ROOT = Path('/kaggle/working/trackb_r07')\nDEVICE = 'cuda:0'\n\nmanifests = []\nfor p in sorted(INPUT_ROOT.glob('**/TRACKB_INPUT_MANIFEST.json')):\n    obj = json.loads(p.read_text(encoding='utf-8'))\n    manifests.append((obj.get('role'), p, obj))\nroles = [r for r, _, _ in manifests]\nprint('Track-B input roles:', roles)\nrequired = {'core', 'historical_compare', 'gvlid_v5', 'irish_potato'}\nif set(roles) != required:\n    raise RuntimeError(f'Expected exactly {sorted(required)}, got {sorted(roles)}')\ncore_path, core = next((p, o) for r, p, o in manifests if r == 'core')\nrepo_rel = str(core.get('repository_root', '')).strip()\nif not repo_rel:\n    raise RuntimeError('core TRACKB_INPUT_MANIFEST.json must define repository_root')\nREPO_ROOT = (core_path.parent / repo_rel).resolve()\nRUNNER = REPO_ROOT / 'journal_extension/scripts/run_trackb_r07.py'\nif not RUNNER.is_file():\n    raise RuntimeError(f'Track-B runner missing: {RUNNER}')\n\nversions = {\n    'torch': metadata.version('torch'),\n    'torchvision': metadata.version('torchvision'),\n    'numpy': metadata.version('numpy'),\n    'Pillow': metadata.version('Pillow'),\n    'opencv-python-headless': metadata.version('opencv-python-headless'),\n}\nexpected = {\n    'torch': '2.12.1',\n    'torchvision': '0.27.1',\n    'numpy': '2.5.2',\n    'Pillow': '12.3.0',\n    'opencv-python-headless': '4.12.0.88',\n}\nprint('Locked package versions:', versions)\nif versions != expected:\n    raise RuntimeError(f'Dependency lock mismatch. Expected {expected}, got {versions}. Configure Kaggle Dependency Manager before Save & Run All.')\n\nprint('Repository root:', REPO_ROOT)\nprint('Working free GB:', round(shutil.disk_usage('/kaggle/working').free / 1024**3, 2))\nsubprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], check=True)\n"""),
        md("""## Execute the frozen Track-B controller\n\nThis is intentionally one clean process. The runner will refuse to instantiate an R07 classifier on external images until **both** candidate audits are terminal and their seal self-hashes verify."""),
        code("""if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):\n    raise RuntimeError(f'Output directory must be empty for a clean claim run: {OUTPUT_ROOT}')\ncmd = [\n    sys.executable, str(RUNNER),\n    '--input-root', str(INPUT_ROOT),\n    '--output-root', str(OUTPUT_ROOT),\n    '--device', DEVICE,\n    '--workers', '4',\n    '--mode', 'all',\n]\nprint('Launching:', ' '.join(cmd))\nsubprocess.run(cmd, cwd=REPO_ROOT, check=True)\n"""),
        md("""## Terminal evidence\n\nThe next cell reads only persisted artifacts written by the controller. A valid run must end with independent QA `PASS` and `TRACK_B_CLOSED`."""),
        code("""qa = json.loads((OUTPUT_ROOT / 'TRACKB_FINAL_QA.json').read_text(encoding='utf-8'))\nclosure = json.loads((OUTPUT_ROOT / 'TRACKB_FINAL_CLOSURE.json').read_text(encoding='utf-8'))\npackages = json.loads((OUTPUT_ROOT / 'TRACKB_PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))\nif qa.get('status') != 'PASS':\n    raise RuntimeError('Track-B independent QA did not PASS')\nif closure.get('status') != 'TRACK_B_CLOSED':\n    raise RuntimeError('Track B did not reach TRACK_B_CLOSED')\nprint(json.dumps(closure, indent=2, sort_keys=True))\nprint(json.dumps(packages, indent=2, sort_keys=True))\n"""),
        md("""## Interpretation boundary\n\n`EXT-I` supports only audit-bounded source-independent wording for the candidate's frozen mapped scope. `EXT-S` supports cross-dataset/external-domain stress-test wording only. `EXT-X` produces no claim-making classifier inference. No Track-B outcome establishes 120-class field generalization, agronomic diagnosis/treatment readiness, or device/runtime performance."""),
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
