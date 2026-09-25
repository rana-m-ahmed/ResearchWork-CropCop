# Prompt 1 Kaggle execution runbook

## Notebook

Use `CropCop_Prompt1_R07_Foundation_QA_Locked.ipynb` with a Kaggle T4 GPU and Internet enabled. The notebook fetches the immutable research base commit `a4f9d6e3d417b024ecbfbf4247cd847881935236`, applies the embedded QA-locked Prompt 1 overlay, consumes the two attached Kaggle datasets, verifies their hashes, installs the pinned runtime, runs Prompt 1, and builds the final bundle. No ZIP upload or dataset download is required.

The notebook executes Prompt 1 only. It does not perform Prompt 2 mobile migration or any claim-producing Track-C phone measurement.

## Kaggle inputs

Attach these two existing Kaggle datasets to the notebook:

- `ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1` — image-backed Final-V1 authority with `final_manifest.csv`, `class_to_idx.json`, and paired `dataset/train/` and `dataset/val/` trees;
- `sabahatabbas/sec-je-r07-cnxtt-context-s1-8904b100d223-a01` — S1 checkpoint containing `selected.g00000170.dc7fea2e8db91bf1.ckpt`.

The QA-locked plan, mobile baseline audit, and research executor are embedded or cloned by the notebook; they are not additional Kaggle inputs.

The canonical authority is the image-backed path ending in `CropCop_Final_v1/audit/final_manifest.csv`. The duplicate `CropCop_Final_v1_CERTIFICATION_REPORTS` tree is deliberately excluded, even when it contains byte-identical audit files. The executor contract requires `final_manifest.csv`; a dataset containing only `training_manifest.csv` is not the frozen Final-V1 input and will be rejected.

The test split is neither required nor accessed. If it is present in the attached Final-V1 dataset, the executor still authorizes and opens only manifest rows belonging to `DS-V1-TRAIN` or `DS-V1-VAL`.

The S1 checkpoint dataset is private, so the Kaggle account running the notebook must have read access to `sabahatabbas/sec-je-r07-cnxtt-context-s1-8904b100d223-a01`. The notebook discovers attached datasets under `/kaggle/input` by exact SHA-256 identity, so mount-directory names do not matter.

Required identities:

| Item | SHA-256 |
|---|---|
| `plan.md` | `c2a354cb0f9e5033742deb354b1fc42b49181e8884f5b6dfde14d2d5018a59c0` |
| S1 checkpoint | `dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974` |
| Final-V1 manifest | `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2` |
| Class map | `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2` |
| P0 audit | `d7acc4e615b5edd00fbd66b70a940452935c8cde94bb0d8860802be1fdc07ce5` |
| ExecuTorch Android 1.3.1 AAR | `a9acc1e85a7f8b45c06ecb0fea398ebc6a5ad474d53278c2058c060ba4f37d17` |

The scientific/runtime Python stack is pinned to the frozen S1 environment: Python 3.12.x, PyTorch 2.12.1 with CUDA 13.0 on Tesla T4, torchvision 0.27.1, ExecuTorch 1.3.1, TorchAO 0.17.0, scikit-learn 1.7.1, Pillow 12.3.0, NumPy 2.5.2, and wrapt 1.17.3. It is installed under `/kaggle/working/cropcop_prompt1_venv`; Kaggle's global Python environment is never modified. Because Kaggle's container can disable `ensurepip`, the notebook creates the environment with `--without-pip` and safely seeds pip into that environment from the working system pip. It then runs `pip check` and an exact-version/import probe before preflight. The pinned wrapt package satisfies Kaggle's injected `sitecustomize` instrumentation without modifying the global environment. Source replay uses the original validation micro-batch size of 16. Preflight fails before expensive work if Kaggle assigns a different GPU/runtime family. When PTQ is enabled, preflight also performs a small end-to-end PT2E conversion, Q/DQ coverage, XNNPACK lowering, serialization, runtime-load, and forward probe so toolchain incompatibility is detected before the validation-wide stages.

Known duplicate certification copies of `final_manifest.csv` and `class_to_idx.json` are handled fail-closed: the executor selects the single hash-valid pair structurally bound to real `dataset/train` and `dataset/val` directories.

## Execution

1. Create or open a Kaggle notebook from `CropCop_Prompt1_R07_Foundation_QA_Locked.ipynb`.
2. Add both datasets above using **Add Input**. Select a Tesla T4 GPU and turn Internet on.
3. Ensure the Kaggle account can read the private S1 checkpoint dataset named above.
4. Run all cells in order. The first cells clone/verify/install automatically; no input path edits are needed.
5. Leave `ENABLE_SINGLE_ROUTE_PTQ = True` to execute the prospectively frozen one-route PTQ design.
6. Do not edit FP32 parity thresholds or PTQ acceptance rules after their result cells have run.
7. Download `PROMPT1_KAGGLE_EXECUTION_BUNDLE.zip` from the final cell.
8. Save a Kaggle notebook version to preserve immutable cell output and environment logs.

Start from a fresh Kaggle session when replacing an older notebook version. This removes global-package changes made by earlier revisions and lets the corrected bootstrap replace any checkout based on the wrong `main` branch.

## Long-running stages and resume

The full validation-wide FP32 and INT8 runtime passes use the Android-compatible fixed batch-1 artifact schema and can be long-running. Evidence is written stage by stage under `/kaggle/working/cropcop_prompt1`.

With `RESUME_COMPLETED_STAGES = True`, rerunning the notebook skips a stage only after the executor verifies the marker self-hash, current executor hash, frozen inputs, clean Git commit, prerequisite stages, and referenced artifact hashes. A stale, tampered, failed, or incomplete stage is rerun rather than silently trusted.

Preflight also performs a tiny end-to-end ExecuTorch/XNNPACK export, FlatBuffers serialization, runtime load, and forward execution. This catches an incompatible Kaggle `flatc`, wheel, or native runtime before the large R07 export begins.

## Final output

The final ZIP contains:

- `PROMPT1_FOUNDATION_HANDOFF.json`;
- all P0/F0/F1-F7 evidence and locks;
- the production-safe deployment bundle;
- the separately isolated restricted verification bundle;
- exact artifact/file manifests and SHA-256 identities;
- the exact executor, notebook, and runbook used for the run;
- FP32 artifact and the INT8 artifact only if the one frozen route passes;
- a terminal INT8 failure certificate when PTQ or retention fails.

Validation logits, targets, row identities and calibration manifests are restricted evidence and are never copied into the production bundle.

The final ZIP is reopened before the download link is shown. Every member is checked for a safe unique path, CRC validity, expected membership, byte count, and SHA-256 identity against `FINAL_FILE_MANIFEST.json`.
