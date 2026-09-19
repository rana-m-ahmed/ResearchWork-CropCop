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
        md("""# CropCop Track B — Safe Historical Comparison Builder

**Infrastructure-only notebook; not a claim-producing evaluation.**

This post-closure builder deliberately indexes only the frozen Final-V1 **training + validation** development surface: **92,744 images** (76,376 train + 16,368 validation). It never opens consumed V1-test image bytes.

Because this is a partial historical comparison surface, the resulting package has a hard **EXT-S ceiling**. `EXT-I` remains available only if a complete 117,546-image comparison representation that was created before V1-test closure is genuinely recovered and independently verified.

Attach:
1. exactly one immutable Track-B `core` package;
2. `ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1` as the source of the frozen train/validation images.

Use Kaggle **T4x2**; the builder intentionally uses only `cuda:0`.
"""),
        md("""## Environment

Configure Kaggle Dependency Manager with `journal_extension/track_b_r07/requirements-trackb.lock.txt` before **Save & Run All**.

This notebook creates only hash/feature/geometric comparison material. It generates **no CropCop classifier predictions**, performs **no model selection**, and does **not** access `dataset/test/`.
"""),
        code("""from pathlib import Path
import json, shutil, subprocess, sys

INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working/cropcop_hist_compare_v1')

core = []
for p in INPUT.glob('**/TRACKB_INPUT_MANIFEST.json'):
    obj = json.loads(p.read_text())
    if obj.get('role') == 'core':
        core.append((p, obj))
if len(core) != 1:
    raise RuntimeError(f'Need exactly one Track-B core package; found {len(core)}')
core_p, core_o = core[0]

v1_roots = []
for manifest in INPUT.glob('**/CropCop_Final_v1/audit/final_manifest.csv'):
    cropcop_root = manifest.parent.parent
    image_root = cropcop_root / 'dataset'
    if (image_root / 'train').is_dir() and (image_root / 'val').is_dir():
        v1_roots.append((cropcop_root.resolve(), manifest.resolve(), image_root.resolve()))
unique = {str(row[0]): row for row in v1_roots}
if len(unique) != 1:
    raise RuntimeError(
        'Attach exactly one frozen CropCop Final-v1 source containing dataset/train and dataset/val; '
        f'found {list(unique)}'
    )
cropcop_root, v1_manifest, image_root = next(iter(unique.values()))

repo = (core_p.parent / core_o['repository_root']).resolve()
runner = repo / 'journal_extension/scripts/build_trackb_historical_compare.py'
if not runner.is_file():
    raise RuntimeError(f'Builder missing: {runner}')

def f(bundle_p, bundle_o, key):
    return (bundle_p.parent / bundle_o['files'][key]['path']).resolve()

dino = f(core_p, core_o, 'dino_checkpoint')
factory = f(core_p, core_o, 'dino_factory_manifest')
execution_lock = f(core_p, core_o, 'execution_lock')
code_attestation = f(core_p, core_o, 'code_attestation')
factory_root = (core_p.parent / core_o['dino_factory_source_root']).resolve()

print('Frozen V1 manifest:', v1_manifest)
print('Development image root:', image_root)
print('Builder policy: train + val only; consumed test image bytes remain unopened')
print('Free working GB:', round(shutil.disk_usage('/kaggle/working').free / 1024**3, 2))
subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
    check=True,
)
"""),
        code("""if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise RuntimeError(f'Output must be empty: {OUTPUT}')

cmd = [
    sys.executable, str(runner),
    '--v1-manifest', str(v1_manifest),
    '--image-root', str(image_root),
    '--dino-checkpoint', str(dino),
    '--factory-manifest', str(factory),
    '--factory-source-root', str(factory_root),
    '--repo-root', str(repo),
    '--execution-lock', str(execution_lock),
    '--code-attestation', str(code_attestation),
    '--output-dir', str(OUTPUT),
    '--device', 'cuda:0',
    '--workers', '4',
    '--orb-chunk-size', '512',
    '--dino-batch-size', '64',
]
print('Launching safe development-surface historical index builder')
subprocess.run(cmd, cwd=repo, check=True)
"""),
        code("""manifest = json.loads((OUTPUT / 'TRACKB_INPUT_MANIFEST.json').read_text())
cert = json.loads((OUTPUT / 'BUILD_CERTIFICATE.json').read_text())
expected = {
    'role': 'historical_compare',
    'coverage_scope': 'V1_TRAIN_VAL_ONLY',
    'image_count': 92744,
    'ext_i_eligible': False,
    'maximum_evidence_grade': 'EXT-S',
    'v1_test_image_bytes_accessed': False,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(f'Historical comparison package contract mismatch: {key}={manifest.get(key)!r}')
if cert.get('status') != 'PASS' or cert.get('v1_test_image_bytes_accessed') is not False:
    raise RuntimeError('Historical comparison build certificate did not close safely')

print(json.dumps(cert, indent=2, sort_keys=True))
print(
    'PASS — publish this output directory as an immutable private Kaggle Dataset. '
    'This package is intentionally EXT-S-bounded. Do not attempt to upgrade it by opening V1-test images.'
)
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
    out = Path(__file__).resolve().parents[1] / "kaggle" / "trackb_build_historical_compare.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
