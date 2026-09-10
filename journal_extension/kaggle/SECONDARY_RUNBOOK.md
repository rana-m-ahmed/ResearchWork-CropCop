# CropCop Track-A Secondary — Kaggle Execution Runbook

## Immutable execution source

`8904b100d223e4319776199c87ab397db23600ce`

Repository branch used only as a fetch fallback: `je-tracka-secondary-core-20260910`.

Do not replace the execution SHA with a wrapper/notebook commit.

## Kaggle accounts

- K1: `ranaabdulrehmannn`
- K2: `sabahatabbas`
- K3: `ranamuhammadahmed6`

Every phase that uses the Kaggle API rejects the wrong authenticated Kaggle username.

## Secrets to add to Kaggle

Add `CROPCOP_GITHUB_TOKEN` as a Kaggle Secret on all three accounts. It must be a fine-grained GitHub PAT with access to `rana-m-ahmed/ResearchWork-CropCop` and Contents read/write because evidence branches are published.

Add `KAGGLE_USERNAME` and `KAGGLE_KEY` on K1/K2 and on K3 before Secondary G1/G2. The short K3 Smoke A/B/dual-smoke phases deliberately do not consume the Kaggle API key.

Do not paste secrets into notebook cells. No Hugging Face token is required by this secondary path.

## Global Kaggle settings

- Internet: ON.
- Always qualify/run through **Save Version -> Save & Run All**.
- Interactive Run/Run All is intentionally rejected.
- Do not change Python from the locked Kaggle environment; notebook requires Python 3.12.13.
- Exact training stack is installed from the repository lock:
  torch 2.12.1, torchvision 0.27.1, timm 1.0.26, numpy 2.5.2,
  Pillow 12.3.0, safetensors 0.8.0, kaggle 2.2.4,
  huggingface-hub 1.30.0, transformers 5.0.0.

## Inputs reused from principal execution

### Frozen Final-V1 dataset
Attach the exact same Final-V1 source used for principal execution.
The notebooks locate it by exact hashes, not by display name:

- `audit/final_manifest.csv`: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- `audit/class_to_idx.json`: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`

The notebook refuses a different dataset.

### Frozen principal G1
For K3 Secondary-G1 only, attach:

`ranamuhammadahmed6/cropcop-g1-sealed` — published version `2`

The K3 notebook verifies and safely extracts the package before the secondary sealer consumes it.

## Exact execution order

### 1. K3 — Smoke A
Notebook: `CropCop_K3_Qualification_Controller.ipynb`
Set: `PHASE = "smoke-write"`

Inputs: none.
Accelerator: any CUDA-capable Kaggle GPU; T4x2 is acceptable.
Internet: ON.
Secrets required for this phase: `CROPCOP_GITHUB_TOKEN` only.

Use Save Version -> Save & Run All.

Expected terminal marker: `CROPCOP SMOKE A COMPLETE`, `STATUS=PASS`.

Keep the exact Notebook Output.

### 2. K3 — Smoke B, fresh Saved Version
Same notebook.
Set: `PHASE = "smoke-restore"`.

Attach exactly the Smoke-A Notebook Output from step 1.
Accelerator: CUDA-capable GPU.
Use a fresh Save & Run All.

Expected output includes `SMOKE_B_EVIDENCE.json`.

Keep this Notebook Output.

### 3. K3 — Dual-T4 smoke
Same notebook.
Set: `PHASE = "dual-gpu-smoke"`.

Attach exactly the Smoke-B Notebook Output from step 2.
Accelerator: **T4 x2**.
Use Save & Run All.

Expected output:
`/kaggle/working/cropcop-dual-gpu-smoke/DUAL_GPU_SMOKE_EVIDENCE.json`
with terminal PASS.

Keep this Notebook Output.

### 4. K3 — Secondary G1
Same notebook.
Set: `PHASE = "secondary-g1"`.

Attach:
1. frozen Final-V1 dataset;
2. `ranamuhammadahmed6/cropcop-g1-sealed` version `2`;
3. Smoke-B Notebook Output from step 2;
4. Dual-T4 Smoke Notebook Output from step 3.

Accelerator: **None / CPU**.
Internet: ON.

The notebook creates/versions this private dataset:

`ranamuhammadahmed6/cropcop-secondary-g1-8904b100`

Expected:
- `SECONDARY_G1_MODEL_IDENTITY_SEAL.json`
- `SECONDARY_G1_PUBLICATION_RECEIPT.json`
- `SECONDARY_G1_TERMINAL_EVIDENCE.json`
- round-trip verification PASS.

After PASS, grant K1 (`ranaabdulrehmannn`) and K2 (`sabahatabbas`) read access to this private dataset before their scientific launch. This sharing can be done while K3 proceeds with Secondary G2.

### 5. K3 — Secondary G2
Same notebook.
Set: `PHASE = "secondary-g2"`.

Attach:
1. frozen Final-V1 dataset;
2. new private Secondary-G1 dataset `ranamuhammadahmed6/cropcop-secondary-g1-8904b100`.

Accelerator: **T4 x2**.

The notebook automatically creates private recovery datasets for the four calibration profiles and performs:
- MNV4 direct;
- MNV4 teacher;
- EfficientNet-B0;
- ConvNeXt-Tiny;
- checkpoint save -> Kaggle private dataset -> local deletion -> restore -> resumed progress;
- selected-checkpoint contract probe;
- identity mismatch rejection;
- publication idempotency.

Expected terminal file:

`/kaggle/working/cropcop-secondary-g2/SECONDARY_G2_CALIBRATION_BARRIER.json`

Keep the exact K3 G2 Notebook Output.

### 6. K1 + K2 — launch secondary science in parallel

Only after K3 Secondary-G2 is PASS.

#### K1
Notebook: `CropCop_K1_Mechanism_Controller.ipynb`

Attach:
1. frozen Final-V1 dataset;
2. shared Secondary-G1 private dataset;
3. K3 Secondary-G2 Notebook Output.

Accelerator: **T4 x2**.

Runs:
- GPU0: `R12-MNV4-LOGITS-S1`
- GPU1: `R12-MNV4-FEATURE-S1`

#### K2
Notebook: `CropCop_K2_Context_Controller.ipynb`

Attach the same three inputs.
Accelerator: **T4 x2**.

Runs:
- GPU0: `R06-EFFB0-CONTEXT-S1`
- GPU1: `R07-CNXTT-CONTEXT-S1`

Start K1 and K2 as close together as practical.

## Long-run continuation

The scientific notebooks use private Kaggle datasets, not Notebook Output, as recovery storage.

If a K1/K2 run ends with `CONTINUATION_REQUIRED`:

1. Do not change the notebook.
2. Do not change the source SHA.
3. Do not change the attached Final-V1, Secondary-G1, or G2-barrier inputs.
4. Do not change the Kaggle account.
5. Keep T4 x2.
6. Use **Save Version -> Save & Run All** again.

The parent probes each durable target:
- checkpoint present -> resume is mandatory;
- pristine first version -> fresh start;
- later version with no checkpoint index -> hard fail;
- restore failure -> hard fail.

Completed scientific checkpoints are recognized before the training loop and are not advanced again.

## Optional use of K3 after G2

Once K1/K2 scientific runs are underway, K3 can remain spare for recovery or run the non-blocking S3 principal validation backfill.

K3 controller:
`PHASE = "principal-backfill-s3"`

Input: frozen Final-V1 dataset.
Accelerator: T4 x2.

This does not gate R06/R07/R12 science.

After K1/K2 secondary science is terminal, their controllers can optionally be changed to:
- K1: `principal-backfill-s1`
- K2: `principal-backfill-s2`

## Things not to attach

- No RFDV teacher dataset to K1/K2 science.
- No historical principal training outputs.
- No V1-test dataset.
- No protected external cohort.
- No manually downloaded EfficientNet/ConvNeXt files; Secondary G1 obtains and provenance-checks the frozen official artifacts.
- No old Smoke A/B evidence from the principal source.
- No previous scientific Notebook Output for continuation.

## Stop conditions

Do not proceed to the next gate if any phase:
- is not a Kaggle Batch/Saved-Version run;
- uses a different source SHA;
- reports a dependency-lock mismatch;
- reports wrong Kaggle account;
- sees anything other than exactly T4x2 where T4x2 is required;
- fails a smoke/G1/G2 barrier;
- cannot verify a private durable target;
- reports ambiguous recovery state.

Send the final evidence/output to ChatGPT for independent verification after Smoke B, Dual Smoke, Secondary G1, and Secondary G2.
