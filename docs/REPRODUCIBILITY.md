# Reproducibility guide — EAAI journal extension

> Paper-to-evidence navigation: [`PAPER_EVIDENCE_AUTHORITY.md`](PAPER_EVIDENCE_AUTHORITY.md)

## Reproduction layers

CropCop separates identity, analysis, execution and archival reproducibility rather than collapsing them into a single “code available” claim.

### Repository validation

```bash
python scripts/validate_repository.py --strict
python -m pytest -q
```

### Benchmark identity

- final frozen benchmark: 109,107 rows / 120 classes;
- split: 76,376 train / 16,368 validation / 16,363 historical test;
- class map SHA-256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`;
- frozen manifest SHA-256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`.

The source corpus itself is not redistributed by this repository.

## Track A — selector reproduction

Track A compares R04 MobileNetV4, R06 EfficientNet-B0, R07 ConvNeXt-Tiny and R13 differential-attention ViT under a common train/validation contract. The selector is frozen before downstream external/device evidence and reproduces R07.

No downstream Track-B or Track-C outcome may be used to substitute a different family.

## Track B — external-source evaluation

Retained cohorts:

- GVLiD v5 — DOI `10.17632/wkymf8bhcg.5`;
- Irish Potato v01 — DOI `10.5281/zenodo.8286529`.

All three fixed R07 states are evaluated under native 120-way inference with exactly seven mapped labels. Out-of-mapped-scope predictions remain errors; there is no mapped-logit renormalization.

The accepted Track-B evidence archive is identity-bound at:

`22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3`

The complete private archive is not stored in ordinary public Git. Its exact expected member inventory and fail-closed verifier are preserved in the reviewed Part-2 release package. The verifier performs no inference.

## Track C — exact state reconstruction

Deployment state: `R07-CNXTT-CONTEXT-S1`, selected epoch 27.

Checkpoint SHA-256:

`dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974`

The state is reconstructed as `torchvision.models.convnext_tiny(weights=None, num_classes=120)` and strictly loaded. Validation replay over all 16,368 rows reproduces accuracy, balanced accuracy, macro-F1 and NLL within `1e-6`.

## Quantization/export reproduction

Accepted route: `R07-XNNPACK-PT2E-STATIC-INT8-PC-v1`.

Calibration is DS-V1-TRAIN only with seed `285554146`: rank rows by SHA-256 of `seed|stable_row_id`, take eight per class (960), then the first 64 globally ranked remaining rows for exactly 1,024 unique rows.

Pinned accepted stack: Python 3.12.x, PyTorch 2.12.1, torchvision 0.27.1, ExecuTorch 1.3.1, torchao 0.17.0, scikit-learn 1.7.1, Pillow 12.3.0, NumPy 2.5.2, CUDA 13.0 / Tesla T4.

Artifacts:

- FP32: 111,741,536 B, SHA-256 `61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec`;
- INT8: 28,555,872 B, SHA-256 `2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`.

Produced binaries are not redistributed in ordinary public Git pending licence review.

## Runtime evidence reproduction

Public Track-C evidence supports independent checking of:

- FP32-host vs INT8-host = 252/256; mismatches 11, 27, 155, 255;
- FP32 host vs Pixel FP32 = 256/256;
- INT8 host vs Pixel INT8 = 256/256;
- raw INT8 vs paired canonical INT8 = 254/256; mismatches 11, 255;
- Pixel timing distributions and nonstationarity;
- POCO original-runtime failure;
- POCO compatible-runtime canonical = 255/256 and raw = 253/256.

These comparisons answer different questions and must not be collapsed into one deployment-fidelity score.

## Public/restricted boundary

Restricted assets include the consolidated benchmark images, final R07-S1 checkpoint, produced FP32/INT8 PTEs, raw logits and the complete private Track-B evidence archive. Hashes and reconstruction contracts are published where permitted so authorized reviewers can verify identity.

An immutable release/tag should be created only after the final journal-facing documentation and submission state are frozen.
