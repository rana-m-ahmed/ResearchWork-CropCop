# Modern Backbone Selection Audit — v1.2

**Status:** PRE-EXECUTION ARCHITECTURE FREEZE  
**Date:** 2026-09-13

## Decision

Add exactly one modern architecture: `vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k` as family **R13**.

## Why R13 is the best fit

R13 is a pure Vision Transformer implemented in the already frozen `timm==1.0.26` environment. Its upstream model card reports 22.5M parameters, 6.3 GMACs, 256x256 input and approximately 83.24% ImageNet-1K top-1. It combines register tokens, global average pooling and differential attention, giving the journal study an attention-based architecture that is technically distinct from all three existing direct candidates.

The important methodological advantage is not headline ImageNet accuracy. It is that R13 changes **architecture family** while preserving the same broad transfer-learning regime (ImageNet-1K) and existing software dependency, reducing confounding and implementation drift.

Its upstream pretrained configuration uses mean/std 0.5/0.5 rather than CropCop's common CTC-v2 normalization. R13 therefore enters science only after the frozen analytical patch-embedding normalization-equivalence transform and numerical parity gate in `r13_pretrained_identity_and_normalization_contract_v1_2.json` pass. This preserves one common CTC-v2 data pipeline without quietly handicapping the pretrained ViT.

## Finalist audit

### R13 differential ViT — SELECTED
- pure Transformer;
- native 256 input;
- ImageNet-1K pretraining;
- no new dependency beyond frozen timm;
- no custom CUDA kernel;
- strong contemporary baseline;
- practical parameter scale;
- exact upstream safetensors object can be SHA-bound;
- cleanest fit to the existing training/checkpoint kernel.

### CARE-S2 — NOT SELECTED
CVPR 2025 mobile-friendly linear-attention Transformer, approximately 19.5M parameters / 1.9 GMACs / 82.1% ImageNet top-1 in the authors' report. Scientifically attractive, but it requires an external implementation/checkpoint stack and separate compatibility qualification. R13 gives comparable modern attention diversity while remaining inside the locked training software.

### MambaVision-T — NOT SELECTED
CVPR 2025 hybrid Mamba-Transformer, approximately 31.8M parameters / 4.4 GFLOPs / 82.3% ImageNet top-1 in the authors' report. Not selected because the reference implementation depends on Mamba selective-scan/custom-kernel machinery and introduces a materially different runtime/export risk. Upstream source/weight licensing also differs from the existing stack. Adding it would increase engineering surface more than scientific value for this study.

### TinyNeXt-M — NOT SELECTED
ICCV 2025 hybrid ViT for TinyML. Excellent efficiency, but its reported ImageNet top-1 is substantially below the high-capacity reference role we want from the single modern comparator. It is less useful here because CropCop already contains mobile-focused candidates.

### SHViT-S4 — NOT SELECTED
Strong mobile Transformer and available in timm, but the 2024 architecture and lower reported ImageNet top-1 make it less compelling than R13 as the final contemporary reference.

### RepViT-M1.5 / MambaOut-Tiny — NOT SELECTED
Both are modern and strong, and both integrate cleanly through timm, but they remain convolution/gated-CNN families. They would add less architectural diversity because the current pool already contains three convolution-dominant families.

### SwinV2-T — NOT SELECTED
Extremely reproducible through torchvision and native 256, but older and less representative of the most recent attention designs than R13.

### TransNeXt — NOT SELECTED
Strong robust Transformer results, but the reference implementation uses a custom CUDA extension and is from 2024. R13 avoids that compatibility surface.

### DINOv3 / EUPE / other large-scale foundation encoders — EXCLUDED FROM PRIMARY POOL
These are important contemporary models, but their substantially larger web-scale/self-supervised pretraining changes both the model and the pretraining-data regime. A win would therefore not be interpretable as an architecture comparison against the ImageNet-pretrained candidates. They may be cited as related work, but they are not allowed into the primary selection pool under this amendment.

## Anti-fishing conclusion

The architecture pool closes after R13. No additional backbone is authorized after any A1/A1.1/A1.2 model result is observed. A later architecture requires a separately justified future study, not another amendment to rescue an unfavorable result.
