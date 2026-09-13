# 03B.2 — Track-A Strengthening Amendment v1.2: Modern ViT Supplement

**Status:** `POST-RESULTS AMENDMENT SUPPLEMENT — PRE-EXECUTION LOCK`  
**Date:** 2026-09-13  
**Parent supplement:** `EAAI-JE-TRACKA-STRENGTHENING-A1.1`  
**Scientific outputs observed from A1/A1.1 before this supplement:** `NONE`

## Purpose

This final pre-execution supplement adds exactly one modern pure-Transformer candidate to the journal-primary architecture pool. The purpose is architectural diversity and contemporary benchmarking, not increasing the number of models until one wins.

## Added candidate

**R13 — Differential ViT / register-token baseline**

Frozen pretrained identity:

`vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k`

Rationale:
- native `256 x 256` input, matching CropCop CTC-v2 resolution;
- ImageNet-1K pretraining, avoiding the large-pretraining-data confound of DINOv3/EUPE-style candidates;
- pure ViT family, adding real architectural diversity beyond MobileNetV4, EfficientNet-B0 and ConvNeXt-Tiny;
- differential attention + register token design gives a contemporary Transformer reference;
- available through the already frozen `timm==1.0.26` stack, so no new custom CUDA extension or external training framework is needed;
- approximately 22.5M parameters / 6.3 GMACs in the upstream model card, placing it in a practical comparison range rather than an oversized foundation-model regime;
- Apache-2.0 model-card licensing and ImageNet-1K training provenance.

## New scientific states

Exactly three direct-CE states are added:
- `R13-VIT-DLITTLE-DIFF-CONTEXT-S1` — seed `21270083`;
- `R13-VIT-DLITTLE-DIFF-CONTEXT-S2` — seed `606135704`;
- `R13-VIT-DLITTLE-DIFF-CONTEXT-S3` — seed `1153870846`.

The states use the unchanged CTC-v2 training/evaluation pipeline, 30 epochs, validation-only selection and `1.00 L_CE`. No new hyperparameter search, model family, seed or objective may be added after this supplement.

A new Amendment-G1A provenance object must bind:
- exact `timm==1.0.26` code identity;
- exact upstream pretrained weight bytes + SHA-256;
- exact model identifier above;
- the frozen normalization-equivalence contract that preserves the upstream patch-embedding function under common CTC-v2 normalization;
- deterministic 120-class classifier reset for each of S1/S2/S3.

## Model-selection pool after v1.2

The journal-primary architecture pool is now closed at exactly four direct families:
1. MobileNetV4 direct — S1/S2/S3;
2. EfficientNet-B0 direct — S1/S2/S3;
3. ConvNeXt-Tiny direct — S1/S2/S3;
4. R13 Differential ViT direct — S1/S2/S3.

No fifth architecture may be added because a later result is inconvenient.

R05 remains the teacher-effect comparator. R12 logits/feature remain mechanism ablations. Neither enters the primary architecture pool.

## Reliability / XAI expansion

The fixed corruption, 120-class error, efficiency and Grad-CAM++ analyses expand from 9 to **12 direct architecture states**.

For the ViT, Grad-CAM++ must use a token-aware reshape path rather than pretending a patch-embedding convolution is a late semantic feature map. The exact module path and prefix-token count must be resolved and sealed during implementation qualification before XAI outputs are opened.

## Parallel execution

R13 should use the otherwise-idle qualified T4x2 account when available:
- GPU0: S1;
- GPU1: S2;
- first freed GPU: S3.

If no fourth/idle account is available at launch time, R13 waits for the first qualified account that becomes free. It must not alter the scientific identity or ordering of the already frozen A1 runs.

## Anti-confounding rule

The R13 candidate is selected because it provides a current, pure-Transformer, ImageNet-1K-pretrained, native-256 reference inside the existing software stack. Foundation encoders pretrained on substantially larger private/web-scale corpora are not substituted into the primary architecture pool because they would change both architecture and pretraining regime simultaneously.

# `GO — ADD R13 TO IMPLEMENTATION/QUALIFICATION; KEEP SCIENCE CLOSED UNTIL UPDATED G1A/G2A + EXACT-HEAD CI PASS`
