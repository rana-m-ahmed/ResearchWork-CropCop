# Stage 01A-G1P-v2 — Wave 3 Transformers Compatibility Decision

- **Artifact:** `01A_G1P_V2_03_TRANSFORMERS_COMPATIBILITY_DECISION.md`
- **Date:** 2026-09-06
- **Decision:** LOCK `transformers==5.0.0`
- **Decision basis:** pre-results compatibility/reproducibility only
- **Model-quality evidence used to choose version:** NONE
- **New dependency-lock SHA-256:** `6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

## 1. Archived environment evidence

The archived:

`CropCop_Model_Readiness_QA_REPORTS (1).zip`

contains `environment_and_config.json` recording:

- Python `3.12.13`;
- Transformers `5.0.0`.

The same archive's `foundation_model_info.json` and `qa_summary.json` also identify Transformers `5.0.0`.

This proves that 5.0.0 is a real historical CropCop environment version, but it is not treated as proof that every historical final-reference execution used exactly that version.

## 2. Historical v7 compatibility range

The recovered v7.1 training notebook/trainer allowed the Transformers major-version family:

`>=4.56.0,<6.0`

Therefore exact `5.0.0` is inside the historical allowed range.

## 3. Official versioned Transformers evidence

The official Hugging Face Transformers v5.0.0 DINOv3 documentation exposes:

- `DINOv3ConvNextConfig`;
- `DINOv3ConvNextModel`.

It documents direct construction:

```python
config = DINOv3ConvNextConfig()
model = DINOv3ConvNextModel(config)
```

without loading pretrained weights.

The documented default ConvNeXt geometry is:

- hidden sizes: `[96, 192, 384, 768]`;
- depths: `[3, 3, 9, 3]`.

The documentation states that the defaults yield the tiny-style configuration corresponding to the DINOv3 ConvNeXt-Tiny family.

Transformers v5's migration documentation also requires `huggingface_hub>=1.0.0`, so the JE lock's exact `huggingface-hub==1.30.0` is within that compatibility family.

## 4. Exact candidate matrix

A dedicated exact-head GitHub Actions job was added with:

```text
Python             3.12.13
torch              2.12.1
torchvision        0.27.1
timm               1.0.26
numpy              2.5.2
Pillow             12.3.0
safetensors        0.8.0
kaggle             2.2.4
huggingface-hub    1.30.0
transformers       5.0.0
```

The first invocation failed before dependency resolution because the temporary CI shell command contained an escaping error. That was a test-harness defect only and did not constitute a candidate compatibility failure.

The corrected exact candidate job ran on Actions #143 / run ID `34009605946`.

## 5. Resolver/install result

The corrected candidate job:

- installed all exact versions successfully;
- installed the CPython 3.12 wheels for torch/torchvision;
- resolved Transformers `5.0.0`;
- preserved exact HF Hub `1.30.0`;
- completed `python -m pip check` with:
  `No broken requirements found.`

## 6. Offline DINO constructor probe

With both:

`HF_HUB_OFFLINE=1`

and:

`TRANSFORMERS_OFFLINE=1`

the probe imported:

- `DINOv3ConvNextConfig`;
- `DINOv3ConvNextModel`.

The construction section additionally blocked socket `connect()` calls.

Observed:

- no network attempt;
- default hidden sizes `[96,192,384,768]`;
- default depths `[3,3,9,3]`;
- feature width `768`.

## 7. Historical-wrapper synthetic strict-load probe

The candidate test constructed the historical wrapper shape:

```text
DINOv3ConvNextModel(default tiny config)
→ Dropout(0.10)
→ Linear(768,120)
```

It cloned one complete synthetic state dict, constructed a second independent wrapper offline, and performed:

`load_state_dict(..., strict=True)`

Result:

- missing keys: none;
- unexpected keys: none;
- head shape: `[120,768]`;
- synthetic strict load: PASS.

The actual historical RFDV checkpoint is intentionally not present in CI. Strict real-checkpoint load + EMA equality remains a required later non-qualifying readiness gate.

## 8. Lock amendment

The execution lock is amended before any new JE result to include:

`transformers==5.0.0`

in:

- `journal_extension/requirements-training.lock.txt`;
- `journal_extension/requirements-training.txt`;
- `journal_extension/locks/execution_dependency_lock.json`;
- dependency environment validation.

The deterministic dependency-lock self-hash is now:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

This dependency change means all real qualification evidence for the former `767859...` dependency lock remains historical only.

## 9. Wave-3 verdict

# **PASS — TRANSFORMERS 5.0.0 EXACT COMPATIBILITY VERIFIED AND LOCKED**

Wave 4 may implement the exact offline historical EMA teacher factory.

No alternate Transformers version was selected.
