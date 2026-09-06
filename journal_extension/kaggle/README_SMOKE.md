# Stage 01A-G1P-v2.1 — Active Kaggle Operator Guide

This is the **active** operator handoff. Historical Stage-01A-SR/MGPU-QA1 reports remain evidence chronology, not current execution instructions.

Final generator-authorized Stage-01A-G1P-v2.1 execution source:

`3c71331494b3e031bbbbc3f08d27cd2605c31097`

Dependency lock:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

Always verify the literal `AUTHORIZED_SOURCE_SHA` in `generate_canonical_notebook.py` before a real run. Wrapper commits are not execution sources.

## Required pre-qualification readiness

Repository closure does **not** authorize G1 yet.

Before final-source Smoke qualification, run the separate non-qualifying CPU readiness entrypoint against:

- `CropCop-Model-RFDV`;
- the frozen Final-V1 source;
- the pre-created private G1 Kaggle Dataset target.

The readiness entrypoint is:

`journal_extension/scripts/validate_g1_inputs.py`

It must return `G1_INPUT_READINESS.json` with PASS while explicitly reporting no G1 seal, no pair initialization, no training, no optimizer steps, and no V1-test access. Readiness is not a canonical qualification phase.

## Required qualification chronology

After independent readiness PASS:

```text
smoke-write
→ fresh smoke-restore
→ independent audit
→ dual-gpu-smoke
→ independent audit
→ g1
→ calibration-dual
→ principal-dual
```

No G1 is authorized before independently audited final-source Smoke A/B and terminal dual-GPU-smoke evidence.

## Phase requirements

### `smoke-write`

Use clean Kaggle **Save Version → Save & Run All / Batch** execution. Do not press Run/Run All in the editor for qualification.

Only secret required:

- `CROPCOP_GITHUB_TOKEN`

Do not provide Kaggle API credentials or CropCop scientific data/model artifacts.

### `smoke-restore`

Use a **fresh Saved Version**, attach the exact successful Smoke-A Notebook Output, and set:

```text
CROPCOP_SMOKE_A_INPUT_ROOT=/kaggle/input/<exact-smoke-a-output>
```

The wrapper searches only below that explicit root.

### `dual-gpu-smoke`

After independent Smoke A/B audit, attach the exact successful Smoke-B Notebook Output and set:

```text
CROPCOP_SMOKE_B_INPUT_ROOT=/kaggle/input/<exact-smoke-b-output>
```

Run this phase on Kaggle T4×2. It remains synthetic and technical only: no CropCop data, no G1, no G2, no R04/R05.

### `g1`

Only after independent final-source dual-smoke audit. G1 is CPU-defined and does **not** require a T4.

Attach both exact qualification outputs and set:

```text
CROPCOP_SMOKE_B_INPUT_ROOT=/kaggle/input/<exact-smoke-b-output>
CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT=/kaggle/input/<exact-dual-gpu-smoke-output>
CROPCOP_RFDV_ROOT=/kaggle/input/<cropcop-model-rfdv>
CROPCOP_FINAL_V1_ROOT=/kaggle/input/<frozen-final-v1-root>
CROPCOP_G1_PRIVATE_DATASET_SLUG=<kaggle-owner>/<pre-created-private-g1-dataset>
```

Required secrets:

- `CROPCOP_GITHUB_TOKEN`;
- `KAGGLE_USERNAME`;
- `KAGGLE_KEY`.

The frozen source automatically resolves the exact historical teacher path, frozen manifest/class map, source-owned teacher factory/lineage, and official MobileNetV4 pretrained object. Do not supply arbitrary teacher/factory combinations.

### `calibration-dual`

Run on Kaggle T4×2. Attach the sealed private G1 Dataset and set:

```text
CROPCOP_G1_INPUT_ROOT=/kaggle/input/<sealed-g1-dataset>
```

The parent requires exact `G1_PACKAGE.tar` + `G1_PACKAGE_MANIFEST.json`, verifies and safe-extracts them, then executes the complete G1 barrier before GPU child launch. Do not attach RFDV downstream after G1.

Terminal Smoke-B + dual-smoke evidence, production V1 training/validation inputs, calibration inputs, and durable-store configuration remain required by the frozen execution source.

### `principal-dual`

Run on Kaggle T4×2 with the same sealed `CROPCOP_G1_INPUT_ROOT` contract and completed G2 evidence.

Set `CROPCOP_PRINCIPAL_ENVELOPE` to exactly `P1`, `P2`, or `P3`. The repository envelope configs own the fixed pair mapping.

## Batch qualification

Interactive runs are diagnostic only. They cannot become terminal qualifying evidence.

## Prohibited shortcuts

- do not search all of `/kaggle/input`;
- do not bypass readiness, Smoke-B, or dual-smoke evidence;
- do not edit scientific configs/batch semantics;
- do not provide CropCop scientific artifacts for Smoke A/B/dual-smoke;
- do not substitute a different teacher, class order, EMA policy, or MNV4 state;
- do not attach RFDV as a downstream G2/principal dependency after G1;
- do not treat any historical pre-G1P qualification as final-source qualification.
