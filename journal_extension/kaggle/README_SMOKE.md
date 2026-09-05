# Stage 01A-MGPU-QA1 — Active Kaggle Operator Guide

This is the **active** operator handoff. Historical Stage-01A-SR reports are not execution instructions.

Final generator-authorized QA1 execution source:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

This is the post-Smoke-A-debug execution source proven by exact-head source CI. Always verify the literal `AUTHORIZED_SOURCE_SHA` in `generate_canonical_notebook.py` before a real run.

## Required chronology

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

No G1 is authorized before independently audited terminal dual-GPU-smoke evidence.

## Phase requirements

### `smoke-write`

Use clean Kaggle **Save Version → Save & Run All / Batch** execution with a GPU accelerator. Do not press Run/Run All in the editor for qualification; the wrapper now refuses `Interactive` before secrets, clone, or package installation.

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

The wrapper requires exactly one `SMOKE_B_EVIDENCE.json` below that root and exports `CROPCOP_INFRA_SMOKE_EVIDENCE`.

This phase remains synthetic and technical only: no CropCop data, no G1, no G2, no R04/R05.

### `g1`

Only after independent dual-smoke audit. Attach both exact successful outputs and set:

```text
CROPCOP_SMOKE_B_INPUT_ROOT=/kaggle/input/<exact-smoke-b-output>
CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT=/kaggle/input/<exact-dual-gpu-smoke-output>
```

The wrapper requires exactly one expected evidence JSON below each explicit root.

G1/later production routes preserve the frozen execution implementation's real artifact variables and require:
- `CROPCOP_GITHUB_TOKEN`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- the required G1 model/data/teacher artifact paths

### `calibration-dual`

Requires terminal Smoke-B + dual-smoke evidence, completed G1 bundle, production durable-store configuration, and the execution code's required calibration inputs.

### `principal-dual`

Set `CROPCOP_PRINCIPAL_ENVELOPE` to exactly `P1`, `P2`, or `P3`. The repository envelope configs own the fixed pair mapping.

## Batch qualification

Interactive runs are diagnostic only. They cannot become terminal qualifying evidence.

## Prohibited shortcuts

- do not search all of `/kaggle/input`;
- do not bypass Smoke-B or dual-smoke evidence;
- do not edit scientific configs/batch semantics;
- do not provide CropCop scientific artifacts for Smoke A/B/dual-smoke;
- do not treat historical Interactive Smoke A as current qualification.
