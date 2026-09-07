# Stage 01A-G1P-v2 — Wave 2 Historical Teacher Forensics

- **Artifact:** `01A_G1P_V2_02_HISTORICAL_TEACHER_FORENSICS.md`
- **Date:** 2026-09-06
- **Scope:** historical source/evidence reconstruction only
- **Teacher checkpoint regenerated:** NO
- **Real G1 executed:** NO
- **V1 test inference executed:** NO
- **Protected model bytes committed:** NO

## 1. Historical artifacts independently inspected

From the user's archived CropCop materials:

1. `CropCop_Definitive_Training_QA_Production_v7_1_DDP_MAXIMUM_ROBUST.ipynb`
2. its embedded `cropcop_ddp_trainer_v7.py`
3. `CropCop_Final_Model_QA_Evidence_Bundle (1).zip`
4. embedded final QA engine `artifacts/cropcop_final_qa_engine_v1.py`
5. `CropCop_Mobile_Production_v2_0_Evidence (1).zip`
6. frozen CropCop sourcebook/paper evidence
7. current Stage-03R authority/lock held by the repository

Only already-archived metadata/source/evidence was inspected. No new protected dataset inference was performed.

## 2. Historical source hashes

### v7.1 notebook

Independently recomputed SHA-256:

`e116528e7307a29fc2f59a6973e64597bc9b074b35be48bab3959de49dca5bcf`

### Embedded v7 trainer

The notebook contains a compressed/base85-encoded trainer payload.

It was decoded/decompressed independently and hashed.

Recomputed SHA-256:

`a1910fb38f3f7114b8ec607b5ed0408ddaf6abda39b8c29e8819e79b2259b83c`

Decoded trainer bytes:

`60393`

This exactly confirms the historical-trainer candidate in the independent audit.

### Final QA engine

From:

`CropCop_Final_Model_QA_Evidence_Bundle (1).zip/artifacts/cropcop_final_qa_engine_v1.py`

independently recomputed SHA-256:

`9b82b14303a6abb90ac64914b04f44a6d63ee56b12bc4c5a58957356a2ba8786`

This exactly confirms the final-QA-engine candidate.

## 3. Historical dataset/model locator

Archived final-QA sidecar evidence records:

```text
/kaggle/input/datasets/ranamuhammadahmed6/cropcop-model-rfdv/
cropcop_runs/stage1_dino_tiny_ce_256/checkpoints/best_macro_f1.pt.meta.json
```

Therefore the historical Kaggle dataset locator is:

`cropcop-model-rfdv`

and the exact teacher checkpoint relative path is:

`cropcop_runs/stage1_dino_tiny_ce_256/checkpoints/best_macro_f1.pt`

The real checkpoint binary is intentionally not copied into public Git.

The later non-qualifying G1 readiness job must mount the historical RFDV dataset and re-hash the actual bytes at this exact path.

## 4. Exact teacher checkpoint identity

The following independent historical evidence all bind the same reference checkpoint SHA:

- final reference-model card;
- final model certification report;
- final QA cache metadata;
- later MobileNetV4 artifact/deployment lineage;
- current Stage-03R scientific authority.

Frozen SHA-256:

`74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`

No contradictory teacher SHA was found.

Substitution remains forbidden.

## 5. Exact historical model identity

The historical v7 trainer model specification binds:

```text
model_key = dino_tiny
model_id  = facebook/dinov3-convnext-tiny-pretrain-lvd1689m
dropout   = 0.10
```

The trainer's wrapper is:

```text
backbone
→ backbone.pooler_output
→ Dropout(0.10)
→ head = Linear(hidden, 120)
```

The final QA engine independently reconstructs the same wrapper.

Historical experiment identity:

`stage1_dino_tiny_ce_256`

Resolution:

`256`

Classes:

`120`

## 6. Canonical historical state is EMA

This is established mechanically, not inferred from a paper sentence.

The historical v7 trainer's `ModelEMA` constructor enumerates:

`model.state_dict()`

and stores every tensor satisfying:

`torch.is_floating_point(v)`

Its serialized state is:

```text
ema = {
  decay,
  shadow = {state_key: floating_tensor}
}
```

The certified final QA loader then:

1. strict-loads `checkpoint["model"]`;
2. computes all floating keys in the reconstructed model state;
3. requires:

`set(ema.shadow) == floating_state_keys`

4. rejects missing or extra EMA keys;
5. copies every EMA shadow tensor into the corresponding model-state tensor;
6. evaluates that state.

The final reference-model card explicitly labels:

`Canonical weights: EMA`

and validation reproduction evidence confirms the certified loader used the EMA state.

Therefore:

# **canonical historical teacher state = EMA**

A raw-checkpoint-model fallback is not historical-equivalent and is not authorized.

## 7. Checkpoint structural identity

The certified QA engine requires the mounted historical checkpoint to contain at minimum:

- `model`;
- `config`;
- `preprocess`;
- `training_engine_abi`;
- `manifest_file_sha256`;
- `class_map_sha256`;
- `dataset_manifest_fingerprint`;
- `best_metric`;
- `best_epoch`.

It also requires:

```text
config.experiment_id = stage1_dino_tiny_ce_256
config.model_key     = dino_tiny
config.resolution    = 256
```

and rejects `module.*`-prefixed state.

Frozen lineage values are:

- manifest SHA:
  `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class-map SHA:
  `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- retained semantic fingerprint:
  `7c368e6e3d8be3bb3a9a3a5f961075d4faa125bcac2e98a3b55e1a1c61f1c523`

## 8. Classifier key and feature width

The historical wrapper defines:

`self.head = nn.Linear(hidden, num_classes)`

Therefore its classifier weight state key is:

`head.weight`

The certified validation-feature cache metadata records canonical EMA features with shape:

`[3000, 768]`

under builder:

`offline_DINOv3ConvNextConfig_default`

Thus historical feature width is:

`768`

and with 120 classes the expected classifier weight geometry is:

`[120, 768]`

The real mounted checkpoint readiness probe must mechanically re-check this exact state key and shape.

## 9. Class-index semantics

The certified final QA engine loads the exact frozen `class_to_idx.json`, verifies that its values are exactly:

`0..119`

and maps each manifest label directly using:

`df["label"].map(class_to_idx)`

to the model target index.

The historical DINO wrapper returns the 120 head logits directly.

No index-remapping/permutation layer exists in the historical forward path.

Later MobileNetV4 deployment lineage independently records:

- output semantic: raw class logits;
- class-index contract: `class_to_idx.json`;
- the same frozen teacher SHA.

Therefore the strongest evidence-supported historical semantics are:

`historical_index_semantics = class_map_index`

and:

`output_order_transform = none`

This conclusion is not based on classifier shape alone.

## 10. Transformers historical compatibility evidence

The v7.1 historical environment policy allowed:

`transformers>=4.56.0,<6.0`

The final QA engine already contains a preferred offline constructor using:

```python
DINOv3ConvNextConfig()
DINOv3ConvNextModel(config)
```

and archived canonical validation-feature metadata records successful use of:

`offline_DINOv3ConvNextConfig_default`

with no offline-builder error.

This supports investigating the independently proposed exact `transformers==5.0.0` candidate in Wave 3, but does not by itself authorize that new exact lock.

## 11. Historical MobileNetV4 lineage connection

The archived MobileNetV4 production evidence binds its teacher identity to the same exact SHA:

`74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`

and preserves the same frozen manifest/class-map identities.

Therefore the teacher being reconstructed for G1 is the same historical reference used by the documented compact-model lineage; no alternate teacher identity appears.

## 12. Contradiction search

No contradiction was found for:

- RFDV dataset locator;
- exact checkpoint relative path;
- teacher SHA;
- model identifier;
- experiment identifier;
- 256 resolution;
- 120 classes;
- canonical EMA state;
- classifier key `head.weight`;
- 768-dimensional pre-classifier feature;
- direct class-map-index semantics;
- no output permutation.

The actual historical teacher binary still must be byte-verified from the mounted RFDV dataset during the later non-qualifying readiness job.

## 13. Wave-2 verdict

# **PASS — HISTORICAL TEACHER IDENTITY AND CANONICAL EMA SEMANTICS RECONSTRUCTED WITHOUT CONTRADICTION**

Wave 3 may now decide the exact Transformers compatibility lock.

No teacher substitute, raw-state fallback or class permutation is authorized.
