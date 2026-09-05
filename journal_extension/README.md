# CropCop EAAI Journal Extension — Lean JE Core

This namespace implements the minimum Stage-03R-bound machinery required before R04/R05 training.
It is intentionally fail-closed: restricted dataset/model bytes are supplied at execution time and
are verified before use; V1 test and sealed external surfaces are not resolvable by the training API.

## Authority

- Scientific lock: `EAAI-JE-SDL-v2.1-QA`
- Lock SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- Repository baseline at Stage 00: `32190dd86293caa82170df3feea505e3c7443b4b`

## Restricted inputs

Scientific jobs require explicit paths to restricted bytes. Nothing here downloads or guesses them.

Required for R04:
- full frozen V1 manifest whose SHA-256 is `bdb82211...f68e2`;
- frozen 120-class map whose SHA-256 is `46f78117...688d2`;
- V1 image root;
- exact MobileNetV4 ImageNet pretrained object from `timm==1.0.26`, hashed at G1;
- a per-seed paired student initialization produced once and reused by matching R04/R05.

R05 additionally requires:
- historical DINOv3 ConvNeXt-Tiny checkpoint with SHA-256 `74b4701b...3b79`;
- a trusted restricted-environment teacher factory/adaptor. The public JE core deliberately does not
  guess a checkpoint constructor.

## Manifest adapter

The full restricted manifest schema is not reconstructed in public Git. Execution therefore passes
the exact column names explicitly (`stable_row_id`, path, split, class index). The loader validates
the manifest hash before reading rows and authorizes only `DS-V1-TRAIN` and `DS-V1-VAL`.

## Principal flow

1. `python -m cropcop_je.validate --repo-root .`
2. Verify exact pretrained/teacher bytes with `scripts/verify_artifact.py`.
3. Create `PAIR_INIT_S1/S2/S3` with `scripts/prepare_pair_init.py`.
4. Run direct/teacher calibration with `scripts/calibrate.py`.
5. Only after G0/G1/G2 pass, launch the six principal configs.

Calibration outputs are scheduling evidence only and must never become scientific checkpoints.
