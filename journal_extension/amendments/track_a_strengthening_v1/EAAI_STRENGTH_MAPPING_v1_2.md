# EAAI Acceptance-Strength Mapping — v1.2 Modern-Backbone Update

**Status:** `PLANNING / PRE-EXECUTION MAPPING`  
**Date:** 2026-09-13  
**Supersedes only the architecture-benchmarking scope of `EAAI_STRENGTH_MAPPING.md`.**

## Final architecture benchmark

The journal-primary candidate pool is closed at four direct architecture families, each evaluated across the same three frozen seeds:

1. MobileNetV4 direct — mobile-focused convolutional/hybrid baseline from the historical preprint;
2. EfficientNet-B0 direct — classical compound-scaled efficient CNN;
3. ConvNeXt-Tiny direct — modern high-capacity convolutional reference;
4. R13 Differential ViT direct — contemporary pure-Transformer reference with register token + differential attention.

This is stronger than adding several fashionable backbones because it covers four meaningfully different architecture regimes while preserving a bounded, interpretable benchmark.

## Why R13 strengthens EAAI fit

R13 closes the clearest remaining benchmarking gap: the prior direct candidate pool was entirely convolution-dominant. The selected R13 is a native-256, ImageNet-1K-pretrained `timm` Vision Transformer with approximately 22.5M parameters and 6.3 GMACs, so it provides a modern attention-based comparator without introducing web-scale foundation pretraining or a new custom-CUDA framework.

The paper must not claim that R13 is inherently superior because it is newer. It enters the exact same evidence funnel as every other direct architecture: three seeds, checkpoint replay, corruption robustness, 120-class failure analysis, efficiency context and the same model-neutral selection gate.

## Updated strength map

| EAAI/reviewer dimension | v1.2 evidence plan | Expected strength | Residual risk |
|---|---|---|---|
| Architecture breadth | 4 direct families x 3 seeds, including a pure modern ViT | **Very strong** | still a bounded benchmark, not an exhaustive model zoo |
| Contemporary AI relevance | late-2025 differential-attention/register-token ViT reference inside frozen timm stack | **Strong** | novelty belongs to evaluation/system science, not invention of R13 |
| Benchmark fairness | ImageNet-1K pretraining across the primary candidate pool; common CTC-v2; R13 normalization-equivalence gate | **Very strong if G1A parity passes** | parity contract must pass before R13 science |
| Reproducibility | exact model ID, timm version, upstream safetensors SHA/bytes, seeded 120-way initialization | **Very strong** | Hugging Face object must be durably sealed before runs |
| Robustness | 5 corruptions x 3 severities x 12 direct states | **Very strong** | synthetic stress != external field validity |
| Explainability | Grad-CAM++ across 12 direct states with token-aware ViT reshape + faithfulness controls | **Moderate-to-strong** | no lesion masks; no localization claim |
| Model selection neutrality | four-family Pareto + frozen lexicographic gate | **Very strong** | do not add a fifth model after results |
| Runtime relevance | Track C follows the eventual Track-A-selected model | **Potentially decisive** | Transformer may prove more difficult to export/deploy; report that honestly |

## Candidates deliberately not added

- Large DINOv3/EUPE-style encoders: excluded because web-scale/self-supervised pretraining would confound architecture and pretraining regime.
- MambaVision/MobileMamba-style backbones: scientifically interesting but add custom selective-scan/runtime and licensing complexity that is unnecessary for the single modern comparator.
- CARE-S2: strong mobile linear-attention candidate, but requires a separate external implementation/checkpoint environment.
- SwinV2/TransNeXt/SHViT: credible alternatives, but R13 provides the best balance of recency, pure-Transformer diversity, native 256 resolution and existing-stack reproducibility.
- Additional CNNs such as RepViT/MambaOut: would add less architectural diversity to the current convolution-heavy pool.

## Final EAAI-facing contribution stack after v1.2

1. audited 120-class benchmark reconstruction and provenance control;
2. prospective three-seed teacher/direct reconstruction;
3. four-family, three-seed model-neutral architecture benchmarking;
4. three-seed distillation-mechanism ablation;
5. 12-state corruption robustness and 120-class failure analysis;
6. reproducible Grad-CAM++ faithfulness/sanity audit;
7. evidence-frozen primary-model selection;
8. independent Track-B external validation if a clean cohort survives;
9. Track-C quantization/export/device validation of the selected journal model;
10. end-to-end checkpoint/evidence/recovery auditability.

The intended manuscript is therefore an AI engineering, verification, benchmarking and deployment study—not a claim that CropCop invented a new Transformer.
