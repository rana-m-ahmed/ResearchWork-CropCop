# 03A — EAAI Baselines / Ablations Results

**Status:** `GO — SECONDARY BASELINES / ABLATIONS EVIDENCE COMPLETE`  
**Integration date:** 2026-09-12  
**Overall Track-A lock state:** `NOT LOCKED — FINAL LOCK REVIEW DEFERRED`  
**Authority:** `EAAI-JE-SDL-v2.1-QA`  
**Secondary scientific source:** `8904b100d223e4319776199c87ab397db23600ce`  
**Secondary source tree:** `93a3b3f8bd2afcbd9d0fbde8521df970935009df`  
**Principal comparison source:** `f171309fc7e9dc22241ecc137ebbb8e4bcdc5433`  
**Manifest SHA-256:** `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`  
**Class-map SHA-256:** `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`  
**Secondary seed:** `21270083`

## 1. Scope and evidentiary boundary

This stage records the four prospectively frozen secondary Track-A states after terminal scientific execution and independent selected-checkpoint validation replay:

- `R12-MNV4-LOGITS-S1` — MobileNetV4 logits distillation, `0.65 CE + 0.35 KD`;
- `R12-MNV4-FEATURE-S1` — MobileNetV4 feature distillation, `0.85 CE + 0.15 feature`;
- `R06-EFFB0-CONTEXT-S1` — EfficientNet-B0 V1 direct-CE context baseline;
- `R07-CNXTT-CONTEXT-S1` — ConvNeXt-Tiny V1 direct-CE context baseline.

All four scientific states are terminal `PASS`. Their immutable selected checkpoints were replayed on the full `DS-V1-VAL` surface in an inference-only evidence pass. Each replay covered exactly **16,368 unique validation rows** and reproduced frozen selected-checkpoint accuracy, balanced accuracy, macro-F1 and NLL with maximum absolute difference **0.0** under the predeclared `1e-6` tolerance.

The evidence pass performed no training, did not relaunch scientific execution, did not advance optimizer state, did not access `DS-V1-TEST-CONSUMED`, did not access protected external surfaces, and did not publish checkpoint material.

This stage is **descriptive**. It does not introduce new seeds, retuning, post-hoc model selection, significance tests, or causal claims beyond the prospectively frozen design.

## 2. Selected validation results

| Experiment | Role | Selected epoch | Accuracy | Balanced accuracy | Macro-F1 | NLL |
|---|---|---:|---:|---:|---:|---:|
| R12 logits | mechanism / logits KD | 30 | 0.9820992180 | 0.9627036057 | 0.9618870303 | 0.1197457905 |
| R12 feature | mechanism / feature KD | 30 | 0.9827101662 | 0.9595018522 | 0.9584971445 | 0.1095503256 |
| R06 EfficientNet-B0 | context baseline | 26 | 0.9850317693 | 0.9675183746 | 0.9669983766 | 0.0847179264 |
| R07 ConvNeXt-Tiny | context baseline | 27 | 0.9853983382 | 0.9666415042 | 0.9681638485 | 0.0962729904 |

## 3. Bounded descriptive contrasts

The R12 mechanism pair shares the same S1 seed and MobileNetV4 family but changes the auxiliary distillation mechanism. On selected-checkpoint validation macro-F1:

- logits KD − feature distillation = **+0.338989 pp**.

This is a **single-seed descriptive contrast only**. It is not a statistical estimate of a population effect and does not authorize a superiority claim.

For the two architecture-context baselines:

- ConvNeXt-Tiny − EfficientNet-B0 = **+0.116547 pp** macro-F1.

This is also descriptive only. R06 and R07 provide architectural context under their frozen implementation profiles; they do not establish a causal ranking of architectures.

For additional orientation against the frozen principal S1 direct MobileNetV4 selected macro-F1 (`0.9611202956`), the secondary S1 differences are:

| Secondary state | Macro-F1 difference vs principal R04-S1 direct (pp) |
|---|---:|
| R12 logits | +0.076673 |
| R12 feature | -0.262315 |
| R06 EfficientNet-B0 | +0.587808 |
| R07 ConvNeXt-Tiny | +0.704355 |

These cross-state differences are context only. They must not be read as matched causal effects because architecture/objective roles differ. The locked principal teacher-effect conclusion in Stage 02 is unchanged.

## 4. Validation-evidence identities

| Experiment | Run ID | Branch head | Selected checkpoint SHA-256 | Validation evidence SHA-256 | Validation predictions SHA-256 |
|---|---|---|---|---|---|
| R12-MNV4-LOGITS-S1 | `JE-R12-MNV4-LOGITS-S1-8904b100d223-A01` | `9540b5ce80da7d280e575992ae892f54981828a1` | `2377007e4f9ccc35c31ff9583ac6196e183bb03b1aa45ed0b8f58779abbc1827` | `4c71b1d5ae44797c0a4763534843c560d688aaeea3a3c8401242a5168d2e53c4` | `8e25252fa4291ad04cfc30438edceec40c7bc1473dd038e3895122ffd62e8a1d` |
| R12-MNV4-FEATURE-S1 | `JE-R12-MNV4-FEATURE-S1-8904b100d223-A01` | `dea06154ba3e79bed20e88015478588466307d66` | `e5d3b691e9f90602a2785dcdbe4c1c40e482a56c7f72741c708504377eea8950` | `0979ff639835883f0651cbd671f5521aada3de4c7328c11fc455d85fcd659fda` | `dbb22f097cde0d3a3af47e4d46d66cb94680ed274da4b21cd56dfd5276741ed4` |
| R06-EFFB0-CONTEXT-S1 | `JE-R06-EFFB0-CONTEXT-S1-8904b100d223-A01` | `43a916ac4738fa2bb979dbd6f5e82be6b0b5e3ad` | `882e1e45e1d18a8ed8168aab266ce5a66a9b57a74e2a8d98acf692207d2210c7` | `96db1a16b2d0ac8cc01537a2f293ad5f5a97ebb13539207012f825e43551a62f` | `3e6badaedaf9de2b74a546cf64c55422baced303d5767e2e8559839bb3dbc7ef` |
| R07-CNXTT-CONTEXT-S1 | `JE-R07-CNXTT-CONTEXT-S1-8904b100d223-A01` | `03f948fb4adf039967056ffb83e648b8bf0d647f` | `dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974` | `4865e12db8bedca0af264ab8def92a1c10fb091bcc1a5fc18c3176491e82a775` | `951fa257d4f8cc1b8cf14d26be7ee8b7d65e3f4fec416cee4cb1b5f49e496375` |

All four remote validation publications were independently re-read after the Kaggle jobs finished. The branch heads matched the terminal notebook reports exactly. Each validation publication commit added only `validation_evidence.json` and `validation_predictions.jsonl`; it did not delete or rewrite the pre-existing `run_record.json`, `metrics.json`, or `segments.jsonl` training evidence.

## 5. Recovery / retry history and scientific continuity

Secondary execution encountered infrastructure and notebook-packaging failures during qualification, continuation, and later evidence replay. Those failures were treated as technical events, not permission to alter the science.

The final evidence state preserves:

- original A01 scientific run IDs;
- frozen source `8904b100d223e4319776199c87ab397db23600ce`;
- seed `21270083`;
- frozen experiment configs/objectives;
- exact selected checkpoint identities;
- owner-bound private durable checkpoint stores;
- the same `DS-V1-VAL` replay surface;
- the predeclared `1e-6` aggregate replay tolerance.

Failed evidence notebook versions did not create new scientific runs. The successful v5 replay restored the already-terminal selected checkpoints and reproduced the frozen selected metrics exactly.

## 6. Interpretation limits

Stage 03A supports only the following bounded statements:

1. all four frozen secondary states executed to terminal scientific completion;
2. their selected-checkpoint validation metrics are reproducible from immutable checkpoint bytes on the complete frozen validation surface;
3. within the single-seed R12 mechanism pair, logits KD had a descriptively higher selected macro-F1 than feature distillation by `0.338989 pp`;
4. the two architecture baselines provide descriptive context, with ConvNeXt-Tiny `0.116547 pp` above EfficientNet-B0 in selected macro-F1 in these specific frozen runs;
5. no inferential or general architecture-superiority claim is supported by these single-seed secondary states.

No new hypothesis test is authorized here.

## 7. Stage gate

- Four secondary scientific states: **PASS / terminal**
- Four selected-checkpoint validation replays: **PASS**
- Validation rows per run: **16,368 unique rows**
- Maximum aggregate replay difference: **0.0**
- Training during replay: **false**
- Optimizer advancement during replay: **false**
- Consumed V1-test access: **false**
- Protected external access: **false**
- Checkpoint material made public: **false**
- Secondary result interpretation: **descriptive / bounded**
- Overall Track-A final lock: **INTENTIONALLY NOT ISSUED IN THIS PASS**

# `GO — SECONDARY BASELINES / ABLATIONS EVIDENCE COMPLETE`

Stage 03A evidence integration is complete. Track A is **ready for final lock review but remains NOT LOCKED** until a separate explicit closure pass.
