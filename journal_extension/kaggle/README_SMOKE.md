# Stage 01A-G1P-v2.2 — Active Kaggle Operator Guide

This is the **active** operator handoff. Historical Stage-01A-SR/MGPU-QA1 reports remain evidence chronology, not current execution instructions.

Final generator-authorized Stage-01A-G1P-v2.2 execution source:

`beabe97d046e071edacdfa1c6933eeb4edb3a588`

Dependency lock:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

Always verify the literal `AUTHORIZED_SOURCE_SHA` in `generate_canonical_notebook.py` before a real run. Wrapper commits are not execution sources.

## Required pre-qualification readiness

Repository closure does **not** authorize G1 yet.

Before final-source Smoke qualification, run the separate non-qualifying CPU readiness entrypoint against:

- `CropCop-Model-RFDV`;
- the frozen Final-V1 source;
- the intended private G1 Kaggle Dataset slug/ownership policy. Target creation is never implicit. If the target already exists, use `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0`; only an explicitly authorized first creation uses `1`.

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
→ independent terminal G1 audit
→ calibration-dual
→ principal-dual
```

No G1 is authorized before independently audited final-source Smoke A/B and terminal dual-GPU-smoke evidence.

Calibration-dual remains blocked until the fresh v2.2 CPU G1 has terminal PASS evidence and that terminal evidence has been independently audited.

All qualification evidence bound to the superseded v2.1 execution source is historical only and cannot satisfy v2.2 qualification.

For the existing private target `ranamuhammadahmed6/cropcop-g1-sealed`, if Kaggle authoritatively reports the dataset present, private, and owned by the authenticated account, use `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0`. Do not recreate or delete it merely because an older-source publication attempt failed.

## Phase requirements

### `smoke-write`

Use clean Kaggle **Save Version → Save & Run All / Batch** execution. Do not press Run/Run All in the editor for qualification.

Set explicitly:

```text
CROPCOP_EXECUTION_PHASE=smoke-write
```

Only secret required:

- `CROPCOP_GITHUB_TOKEN`

Do not provide Kaggle API credentials or CropCop scientific data/model artifacts.

### `smoke-restore`

Use a **fresh Saved Version**, attach the exact successful Smoke-A Notebook Output, and set:

```text
CROPCOP_EXECUTION_PHASE=smoke-restore
CROPCOP_SMOKE_A_INPUT_ROOT=/kaggle/input/<exact-smoke-a-output>
```

The wrapper searches only below that explicit root.

### `dual-gpu-smoke`

After independent Smoke A/B audit, attach the exact successful Smoke-B Notebook Output and set:

```text
CROPCOP_EXECUTION_PHASE=dual-gpu-smoke
CROPCOP_SMOKE_B_INPUT_ROOT=/kaggle/input/<exact-smoke-b-output>
```

Run this phase on Kaggle T4×2. It remains synthetic and technical only: no CropCop data, no G1, no G2, no R04/R05.

### `g1`

Only after independent final-source dual-smoke audit. G1 is CPU-defined and does **not** require a T4.

Attach both exact qualification outputs and set:

```text
CROPCOP_EXECUTION_PHASE=g1
CROPCOP_SMOKE_B_INPUT_ROOT=/kaggle/input/<exact-smoke-b-output>
CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT=/kaggle/input/<exact-dual-gpu-smoke-output>
CROPCOP_RFDV_ROOT=/kaggle/input/datasets/ranamuhammadahmed6/cropcop-model-rfdv
CROPCOP_FINAL_V1_ROOT=/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1
CROPCOP_G1_PRIVATE_DATASET_SLUG=ranamuhammadahmed6/cropcop-g1-sealed
CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0
```

Readiness has already established that `ranamuhammadahmed6/cropcop-g1-sealed` exists, is private, is owned by the authenticated account, and is ready. For the fresh v2.2 G1 qualification use `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0`; do not recreate or delete that target.

`CROPCOP_FINAL_V1_ROOT` means the directory that directly contains `audit/final_manifest.csv` and `audit/class_to_idx.json`. The wrapper accepts only that exact root or one deterministic normalization from an explicitly supplied outer mount to its immediate `CropCop_Final_v1` child. It never recursively searches `/kaggle/input`.

Private-target policy is explicit and fail-closed:

- `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0`: the private target must already exist.
- `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1`: the frozen G1 code may create the minimal private target if it is missing, then re-preflight ownership/status before later publishing the sealed G1 package as a dataset version and round-trip verifying it.

Readiness may defer target creation; an actual G1 run must choose exactly `0` or `1`. There is no silent create default.

When creation is explicitly allowed, the wrapper treats Kaggle dataset creation as asynchronous. It reuses an existing target, creates the minimal private target only when absence is established, and waits until authoritative private metadata is readable, the exact slug appears in the authenticated account's `mine` listing, and dataset status is ready/completed before invoking frozen G1.

Required secrets:

- `CROPCOP_GITHUB_TOKEN`;
- `KAGGLE_USERNAME`;
- `KAGGLE_KEY`.

The frozen source automatically resolves the exact historical teacher path, frozen manifest/class map, source-owned teacher factory/lineage, and official MobileNetV4 pretrained object. Do not supply arbitrary teacher/factory combinations.

### `calibration-dual`

Run as a fresh Kaggle **Save Version → Save & Run All / Batch** job on **T4×2** with Internet enabled.

Attach exactly:

- the exact successful final-source Smoke-B Notebook Output;
- the exact successful final-source dual-GPU-smoke Notebook Output;
- the sealed private G1 Dataset version produced by the final-source G1;
- `ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1`;
- a Kaggle input containing the official TorchVision 0.27.1 `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` file named exactly `convnext_tiny-983f1562.pth`.

Do not attach RFDV downstream after G1.

Set:

```text
CROPCOP_EXECUTION_PHASE=calibration-dual
CROPCOP_LANE=K1

CROPCOP_SMOKE_B_INPUT_ROOT=/kaggle/input/<exact-final-source-smoke-b-output>
CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT=/kaggle/input/<exact-final-source-dual-output>
CROPCOP_G1_INPUT_ROOT=/kaggle/input/<sealed-final-source-g1-dataset>

CROPCOP_MANIFEST=/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1/audit/final_manifest.csv
CROPCOP_CLASS_MAP=/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1/audit/class_to_idx.json
CROPCOP_IMAGE_ROOT=/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1/dataset

CROPCOP_CNXTT_PRETRAINED=/kaggle/input/<official-cnxtt-input>/convnext_tiny-983f1562.pth

CROPCOP_G2_SUMMARIES_DIR=/kaggle/working/cropcop-g2-summaries
CROPCOP_DURABLE_STORE_KIND=kaggle-dataset
CROPCOP_DURABLE_LOCATOR_TEMPLATE=ranamuhammadahmed6/cropcop-je-{run_id_lower}
```

The wrapper now freezes and exports the certified Final-V1 column contract automatically:

```text
record_key
portable_relpath
split
label
```

Numeric targets are derived only from the hash-locked `class_to_idx.json`; there is no operator class-index column.

Before the first G2 run, these three recovery targets must already exist, be **private**, belong to the authenticated Kaggle account, have a settled/ready version, and be readable:

```text
ranamuhammadahmed6/cropcop-je-cal-mnv4-direct
ranamuhammadahmed6/cropcop-je-cal-mnv4-teacher
ranamuhammadahmed6/cropcop-je-cal-cnxtt
```

Required secrets:

- `CROPCOP_GITHUB_TOKEN`;
- `KAGGLE_USERNAME`;
- `KAGGLE_KEY`.

Do not set `CROPCOP_ENVELOPE_INPUT_ROOT` for a first G2 attempt. That variable is reserved for validated continuation/recovery from a prior envelope.

The parent accepts either the raw exact `G1_PACKAGE.tar` + `G1_PACKAGE_MANIFEST.json` transport or Kaggle's archive-expanded `G1_PACKAGE/` + root manifest representation. Expanded transport is accepted only after exact member, byte-count, SHA-256, deterministic reconstructed tar SHA and tar-size verification. It then executes the complete G1 barrier before GPU child launch.

G2 launches the predeclared scheduling-only calibration envelope:

- slot 0: `CAL-MNV4-DIRECT` (200 optimizer steps);
- slot 1: `CAL-MNV4-TEACHER` (100 optimizer steps);
- first freed T4: `CAL-CNXTT` (100 optimizer steps).

The ConvNeXt control is non-scientific scheduling/context calibration and is bound to the official `convnext_tiny-983f1562.pth` identity. G2 does not consume the protected V1 test.

### `principal-dual`

Run on Kaggle T4×2 with the same sealed `CROPCOP_G1_INPUT_ROOT` contract and completed G2 evidence.

Set explicitly:

```text
CROPCOP_EXECUTION_PHASE=principal-dual
CROPCOP_PRINCIPAL_ENVELOPE=P1  # or P2 / P3
```

The principal envelope has no implicit default. The repository envelope configs own the fixed pair mapping.

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
