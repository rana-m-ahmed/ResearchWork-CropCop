# Track A v1.2 — Canonical Pre-Science and Execution Runbook

**Authority:** `EAAI-JE-SDL-v2.1-QA`  
**Scientific interpretation:** one comprehensive, model-neutral, 21-state journal Track-A experiment executed in computational waves.  
**Candidate pool:** R04 MobileNetV4, R06 EfficientNet-B0, R07 ConvNeXt-Tiny, R13 Differential ViT; S1/S2/S3 for each direct family.  
**Remaining training inventory:** exactly 11 states in `cropcop_je.tracka_v12.EXPERIMENT_SPECS`.  
**Protected before Track-A closure:** `DS-V1-TEST-CONSUMED`, Track-B predictions, Track-C journal-candidate runtime outcomes.

This runbook is operational only. It does not alter any frozen scientific protocol, seed, objective, model family, or selection rule.

## 0. Freeze the execution and analysis source

Use one clean repository checkout and bind every downstream qualification, training, replay, robustness, XAI, auxiliary-analysis and closure artifact to the same full commit SHA:

```bash
export CROPCOP_SOURCE_SHA="$(git rev-parse HEAD)"
export CROPCOP_ANALYSIS_SHA="$CROPCOP_SOURCE_SHA"
test "${#CROPCOP_SOURCE_SHA}" -eq 40
test "$CROPCOP_ANALYSIS_SHA" = "$CROPCOP_SOURCE_SHA"
test -z "$(git status --porcelain)"
```

The worktree must be clean apart from designated output roots. Do not create a later scientific or analysis artifact from a different source SHA. Historical training run records retain their original scientific source SHAs; new post-training evidence additionally records `analysis_source_git_commit=$CROPCOP_ANALYSIS_SHA`.

## 1. Obtain the two exact-head CI attestations

Required GitHub Actions artifacts from the same `${CROPCOP_SOURCE_SHA}`:

- `tracka-v12-pre-science-code-attestation.json`
- `tracka-v12-exact-head-lock-runtime-attestation.json`

The code attestation is emitted only after contract/failure-injection tests, canonical historical-lineage tests, G2A private-durability tests, orchestration/recovery tests and both historical science-diff sentinels pass. It hashes the Track-A v1.2 implementation including G1A/G2A qualification, checkpointing/persistence, the historical replay registry, the reused secondary architecture surface and the canonical final GO sealer. The lock/runtime attestation binds the immutable v1.2 protocol lock, exact R13 runtime/XAI interface, candidate claim boundary, and XAI operationalization to the same source SHA.

**Gate:** no G1A/G2A science authorization if either attestation is absent, stale, non-PASS, or source-mismatched.

## 2. Seal G1A model/initialization identity

Run `journal_extension/scripts/seal_tracka_v12_g1a.py` once into a new immutable bundle directory. Required inputs are:

- frozen V1 manifest and 120-way class map;
- historical principal G1 bundle;
- exact EfficientNet-B0 and ConvNeXt-Tiny official pretrained artifacts + provenance evidence;
- exact R13 upstream safetensors;
- frozen dependency lock;
- `${CROPCOP_SOURCE_SHA}` as `--authorized-source-sha`.

Example skeleton:

```bash
python journal_extension/scripts/seal_tracka_v12_g1a.py \
  --repo-root . \
  --authorized-source-sha "$CROPCOP_SOURCE_SHA" \
  --manifest "$V1_MANIFEST" \
  --class-map "$CLASS_MAP" \
  --image-root "$IMAGE_ROOT" \
  --principal-g1-bundle "$PRINCIPAL_G1" \
  --effb0-pretrained "$EFFB0_PRETRAINED" \
  --effb0-provenance "$EFFB0_PROVENANCE" \
  --cnxtt-pretrained "$CNXTT_PRETRAINED" \
  --cnxtt-provenance "$CNXTT_PROVENANCE" \
  --r13-pretrained "$R13_PRETRAINED" \
  --bundle-dir "$TRACKA_G1A_BUNDLE"
```

**Required output:** `$TRACKA_G1A_BUNDLE/TRACKA_V12_G1A_SEAL.json` with terminal `PASS` and `science_authorized=false`.

G1A must seal:

- exact R12 S2/S3 reused MobileNetV4 pair bytes;
- exact historical teacher bytes/evidence;
- deterministic R06/R07 S2/S3 initializations;
- exact R13 upstream bytes;
- R13 normalization-equivalence parity;
- deterministic R13 S1/S2/S3 initializations;
- dependency-lock identity;
- exact source SHA.

## 3. Run five non-scientific G2A v1.2.2 calibration profiles

Use `journal_extension/scripts/qualify_tracka_v12_profile_v122.py` for exactly these profiles:

1. `CAL-EFFB0`
2. `CAL-CNXTT`
3. `CAL-MNV4-LOGITS`
4. `CAL-MNV4-FEATURE`
5. `CAL-R13`

Each profile must use its frozen representative experiment, one visible T4, calibration mode, initial checkpointing, destructive local checkpoint removal, durable restore, and resumed optimizer progress. Validation metrics must remain disabled and no scientific output may be produced.

**Durability is mandatory and production-equivalent:**

- use `--durable-required`;
- use exactly `--durable-store-kind kaggle-dataset`;
- use one pre-created **private** Kaggle dataset locator in `owner/dataset` form;
- the locator owner must match the authenticated Kaggle account running that profile;
- authenticated read/private-metadata preflight must PASS before calibration;
- the first real calibration sync is the write exercise;
- use **five distinct private Kaggle datasets** across the five profiles. No two calibration profiles may share a durable locator.

The five profiles may run concurrently when five T4 slots and five isolated private durability targets are available.

Every calibration summary must bind the same:

- source Git SHA;
- software-stack SHA;
- G1A seal SHA;
- dependency-lock SHA;
- T4 hardware class.

Each summary must additionally record its Kaggle-private durability backend, locator, successful private-target preflight, successful sync, destructive local deletion, restore and resumed optimizer advance.

## 4. Seal the G2A v1.2.2 barrier and six-slot schedule

```bash
python journal_extension/scripts/seal_tracka_v12_g2a.py \
  --summary "$CAL_EFFB0" \
  --summary "$CAL_CNXTT" \
  --summary "$CAL_MNV4_LOGITS" \
  --summary "$CAL_MNV4_FEATURE" \
  --summary "$CAL_R13" \
  --barrier-out "$TRACKA_G2A_BARRIER" \
  --scheduler-out "$TRACKA_SCHEDULER"
```

The sealer must produce:

- a `1.2.2` full-run forecast barrier;
- a self-hashed G2A durability contract bound to the exact five calibration-summary hashes;
- exactly five unique private Kaggle durability locators;
- deterministic LPT assignment over exactly six slots: K1/GPU0, K1/GPU1, K2/GPU0, K2/GPU1, K3/GPU0, K3/GPU1;
- an exact one-time partition of all 11 remaining states;
- source/G1A/G2A bindings;
- `science_authorized=false`.

Scientific outcomes must never reorder these queues.

## 5. Seal the final science-GO artifact

Only after Steps 1–4 PASS, use the durability-bound canonical GO sealer:

```bash
python journal_extension/scripts/seal_tracka_v12_science_go_v123.py \
  --repo-root . \
  --code-attestation "$CODE_ATTESTATION" \
  --lock-runtime-attestation "$LOCK_RUNTIME_ATTESTATION" \
  --g1a-seal "$TRACKA_G1A_BUNDLE/TRACKA_V12_G1A_SEAL.json" \
  --g2a-barrier "$TRACKA_G2A_BARRIER" \
  --scheduler-freeze "$TRACKA_SCHEDULER" \
  --output "$TRACKA_SCIENCE_GO"
```

The GO artifact must bind one identical source SHA across code attestation, lock/runtime attestation, G1A, G2A and scheduler, and its G2A evidence binding must contain the sealed durability-contract SHA. The authorization validator rejects GO artifacts without that durability binding. It authorizes only the frozen 11 remaining Track-A training states.

**Do not use the legacy GO command and do not launch scientific training without the durability-bound GO artifact.**

## 6. Prepare durable-state mapping

Create one private durable Kaggle dataset locator per remaining experiment ID. The JSON object must contain exactly the 11 IDs in `EXPERIMENT_SPECS`, with unique `owner/dataset` values. No two scientific states may share a durable locator.

Private checkpoint material, model weights, secrets and console logs are not public evidence.

## 7. Launch K1, K2 and K3

On each qualified Kaggle T4x2 account, invoke `journal_extension/kaggle/run_tracka_v12_account.py` with the same:

- `${CROPCOP_SOURCE_SHA}`;
- frozen manifest/class map/image root;
- G1A bundle;
- G2A barrier;
- scheduler freeze;
- science-GO artifact;
- complete durable-map JSON.

Use `--account-id K1`, `K2`, or `K3` respectively.

The parent performs fail-closed preflight before launching either GPU:

1. clean exact source;
2. T4x2 envelope;
3. exact G1A bundle;
4. G2A ↔ G1A binding;
5. scheduler ↔ G2A/G1A/source binding;
6. science-GO binding, including the G2A durability hash;
7. exact durable-map inventory.

Each account launches one independent single-GPU child per physical T4. `DDP`, `DataParallel`, `FSDP`, and cross-run gradient synchronization are forbidden. Child environments strip Git publication credentials and expose exactly one GPU.

The frozen queue is consumed sequentially per physical slot. Session rollover must restore the same scientific state from its verified durable checkpoint; placement is not part of scientific checkpoint identity.

## 8. Complete the 11 remaining training states

A state is terminal only when its run record is `PASS`, its selected checkpoint is hash-bound, its full frozen 30-epoch schedule is complete, and no protected surface was accessed.

Do not add seeds, architectures, objectives, corruption severities, or post-result tuning.

## 9. Build direct-candidate evidence for all 12 direct states

For R04/R06/R07/R13 × S1/S2/S3, run:

- `run_tracka_v12_direct_evidence.py`
- `run_tracka_v12_xai.py`

Both executors must receive:

```bash
--analysis-source-git-commit "$CROPCOP_ANALYSIS_SHA"
```

and must run from a clean checkout exactly at that SHA. Their outputs preserve the original `source_git_commit` of each scientific training run and separately record the frozen `analysis_source_git_commit`.

Required direct evidence per state:

- hash-verified selected checkpoint;
- exact 16,368-row clean `DS-V1-VAL` replay;
- accuracy, balanced accuracy, macro-F1, NLL;
- all 120 classwise precision/recall/F1/support values;
- directed confusion summary;
- fixed five-corruption × three-severity robustness sweep;
- efficiency identity;
- frozen 240-image Grad-CAM++ audit and sanity checks.

Historical R04 S1/S2/S3 and historical R06/R07 S1 must use their already sealed selected checkpoints and original scientific lineages. R06/R07 S2/S3 and R13 S1/S2/S3 use the new G1A lineage. The canonical historical registry must match the immutable Wave-1/Wave-2 closures. Do not rewrite or substitute historical evidence.

## 10. Build auxiliary evidence

For R05 S1/S2/S3 and R12 logits/feature S1/S2/S3, run `run_tracka_v12_auxiliary_evidence.py` with:

```bash
--analysis-source-git-commit "$CROPCOP_ANALYSIS_SHA"
```

- Historical R05 and R12-S1 must match the exact selected checkpoints sealed in the immutable Wave closures and use the original sealed principal pair-init lineage.
- R12-S2/S3 use exact G1A reused-pair bytes.
- Auxiliary states require selected-checkpoint replay and 120-class evidence, but do not enter the primary architecture selector and do not inherit robustness/XAI requirements.

Then run `seal_tracka_v12_auxiliary_analysis.py` over exact R04/R05/R12 S1/S2/S3 evidence. It must reject any evidence whose analysis SHA differs from the current frozen checkout. The outputs are descriptive paired three-seed analyses; no hypothesis test or multiple-comparison p-value is authorized by this executor.

## 11. Seal direct scientific-primary selection

Run `seal_tracka_v12_selection.py` only after all 12 direct states have complete replay/robustness/classwise/efficiency/XAI evidence. Every direct bundle must share one selected checkpoint per state, preserve its scientific source, and bind the current frozen analysis SHA.

Selection remains the frozen v1.2 rule:

1. Pareto frontier over predictive quality, worst-seed reliability, class-tail reliability, robustness and efficiency;
2. if one non-dominated family remains, select it;
3. otherwise use the fixed lexicographic order;
4. an exact recorded-precision tie produces `CO_PRIMARY_TIE` rather than another Track-A experiment.

This direct-selection artifact **must keep Track B and Track C closed**.

## 12. Seal comprehensive 21-state Track A

Run `seal_tracka_v12_comprehensive_closure.py` with:

- direct-selection artifact;
- auxiliary-analysis artifact;
- immutable `WAVE1_PRINCIPAL_VALIDATION_CLOSURE.json`;
- immutable `WAVE2_SECONDARY_VALIDATION_CLOSURE.json`.

The historical closure inventories must match exactly, including the exact four Wave-2 S1 states; supersets are rejected. The direct and auxiliary closure/analysis source must equal the frozen repository head.

The closure must resolve exactly:

- 12 direct candidate states;
- 3 R05 teacher-comparator states;
- 6 R12 mechanism states;
- 21 unique states total.

If the direct result is `SELECTED`, Track B and Track C may receive only the sealed scientific-primary family. If it is `CO_PRIMARY_TIE`, Track A closes scientifically but both handoffs remain closed until the separately frozen deployment-feasibility tie gate resolves the tie.

## 13. Downstream boundary

Only after the comprehensive Track-A closure:

- **Track B:** external/generalization evaluation of the sealed Track-A scientific primary under the independent-cohort/provenance gate.
- **Track C:** quantization/export/runtime evaluation of the sealed Track-A scientific primary.

Neither Track B nor Track C may rewrite the Track-A scientific selection. If deployment later requires a different deployment-primary model, report scientific-primary and deployment-primary separately under the frozen fallback rule.
