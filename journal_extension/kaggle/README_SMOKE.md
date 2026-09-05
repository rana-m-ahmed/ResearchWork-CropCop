# Stage 01A-SR — Kaggle Smoke Operator Guide

This smoke is **API-free**. It uses Kaggle Saved-Version output persistence and the
**Add Input → Notebook Output Files** UI. It does not require a Kaggle API token, Kaggle username
secret, a private recovery dataset, or any CropCop dataset.

## Before Smoke A

1. Enable a GPU accelerator.
2. Add exactly one Kaggle Secret:
   - `CROPCOP_GITHUB_TOKEN`
   
   For a **fine-grained PAT**, configure it as:
   - Resource owner: `rana-m-ahmed`
   - Repository access: only `ResearchWork-CropCop` (or all repositories if intentionally broader)
   - Repository permission: **Contents — Read and write**
   - Store the raw token value only in Kaggle Secrets: no quotes, spaces or line breaks.
   
   A classic PAT must have private-repository `repo` scope, but a fine-grained PAT is preferred.
3. Open `journal_extension/kaggle/canonical_lane.ipynb`.
4. Leave the frozen execution source unchanged: `939455c2cc8787bb295e073d706c32768820bfab`.
5. Set:
   ```
   EXECUTION_PHASE = "smoke-write"
   LANE = "K1"
   ```
6. Normally keep these defaults:
   ```
   REPO_WORKDIR = "/kaggle/working/cropcop-je"
   OUTPUT_ROOT = "/kaggle/working/cropcop-je-output"
   SYNTHETIC_SMOKE_ROOT = "/kaggle/working/cropcop-smoke-input"
   SMOKE_A_EXPORT_ROOT = "/kaggle/working/cropcop-smoke-a-export"
   SMOKE_B_EXPORT_ROOT = "/kaggle/working/cropcop-smoke-b-export"
   ```
7. Use **Save & Run All**.

Before cloning, the notebook now performs three non-secret-leaking checks:
- GitHub API private-repository authorization;
- Git-over-HTTPS `git ls-remote` read authorization;
- `git push --dry-run` to a dedicated `run-evidence/auth-probe-...` ref to prove write access without creating a branch.

If one fails, use the specific error rather than regenerating checkpoints or changing scientific settings.

A successful Smoke A prints an unmistakable `CROPCOP SMOKE A COMPLETE` block. Preserve that exact
Saved Version. Its output contains `SMOKE_A_MANIFEST.json`, `SMOKE_A_EVIDENCE.json`,
`checkpoint_index.json`, and the content-addressed checkpoint beneath `objects/`.

## Before Smoke B

1. Start a **fresh** Kaggle Saved Version/session.
2. Enable a GPU.
3. Keep the same `CROPCOP_GITHUB_TOKEN` secret.
4. Use **Add Input → Notebook Output Files** and select the exact successful Smoke-A Saved Version.
5. Inspect the mounted input path under `/kaggle/input/...`.
6. Set:
   ```
   EXECUTION_PHASE = "smoke-restore"
   SMOKE_A_INPUT_ROOT = "/kaggle/input/<actual-mounted-smoke-a-output>"
   ```
7. Keep the exact same authorized source SHA and lane.
8. Use **Save & Run All**.

Smoke B refuses to train or create a new recovery checkpoint until the attached A manifest, evidence,
checkpoint bytes, source SHA, dependency-lock SHA, qualification ID, identity digest and optimizer step
have been verified. The attached Smoke-A input is read-only and is fingerprinted before/after recovery.

## Not required for Stage 01A-SR

Do not provide:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- Kaggle API token
- private Kaggle recovery dataset
- V1 manifest/images/test
- class map
- MobileNetV4 pretrained bytes
- historical DINO teacher/factory/lineage
- PlantCity / PlantDoc / PlantVillage
- protected external data

Those belong to later stages.
