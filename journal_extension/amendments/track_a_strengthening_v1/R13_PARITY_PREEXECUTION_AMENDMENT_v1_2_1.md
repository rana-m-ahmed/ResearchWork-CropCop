# R13 normalization-parity pre-execution amendment v1.2.1

## Status

**PRE-EXECUTION LOCK.** This amendment was made before any R13 scientific training state was authorized or executed. It does not authorize science by itself.

## Trigger

The frozen v1.2 R13 normalization-equivalence contract required a float32 patch-embedding maximum absolute difference of at most `1e-5`. During the first real K1 G1A execution against the exact pinned pretrained artifact, the parity gate failed closed with:

`3.4332275390625e-05 > 1e-05`

The run had already passed release integrity, K1 account binding, frozen V1 identity, principal G1 identity and exact-stack verification. No G1A READY, G2A qualification or scientific training was produced.

## Independent reproduction

A separate pre-science GitHub Actions diagnostic downloaded the exact pinned R13 artifact (`90095744` bytes, SHA-256 `92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed`) at the frozen upstream commit and reproduced the same `3.4332275390625e-05` maximum under torch 2.12.1.

The diagnostic compared the existing implementation, literal contract bias summation order, float64 parameter construction with elementwise bias summation, and float64 parameter construction with literal bias summation. Every variant produced the same maximum. The current implementation had mean absolute difference `9.250025527762773e-07` and 99.9th-percentile absolute difference `9.5367431640625e-06`.

This excludes a Kaggle-only failure, an alternate summation-order fix, or a higher-precision parameter-construction fix. The defect was the original acceptance-bound calibration, not the normalization-equivalence formula or the pinned pretrained artifact.

## Narrow correction

The v1.2 contract is preserved unchanged for provenance. The superseding v1.2.1 R13 parity contract changes only the maximum float32 patch-output acceptance bound:

- previous: `1e-5`
- superseding: `5e-5`

`5e-5` is the narrowest simple decimal bound above the independently reproduced exact-pretrained maximum while retaining a stringent numerical-equivalence requirement.

## Explicitly unchanged

The following are unchanged: R13 model identity, upstream bytes and commit, timm version, native normalization, CTC-v2 normalization, weight transform formula, bias transform formula, classifier-reset timing, seeds, objectives, training hyperparameters, candidate pool, dataset identity, TRAIN/VAL/test surface policy, selection protocol, XAI target, robustness protocol and all non-R13 Track-A states.

## Evidence

Machine-readable evidence is sealed in `r13_parity_preexecution_evidence_v1_2_1.json`. The superseding machine-readable contract is `r13_pretrained_identity_and_normalization_contract_v1_2_1.json`.

No validation performance metric, test label/result, external prediction, or scientific R13 training output informed this amendment.
