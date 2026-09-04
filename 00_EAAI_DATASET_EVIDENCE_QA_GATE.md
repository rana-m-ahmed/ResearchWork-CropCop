# 00 — EAAI Dataset Evidence QA Gate

**Project:** CropCop  
**Repository:** `rana-m-ahmed/ResearchWork-CropCop`  
**Target:** Engineering Applications of Artificial Intelligence (EAAI), Elsevier  
**Purpose:** Independent post-integration dataset-evidence verification and final pre-amendment gate  
**Scope lock:** Dataset evidence only. EAAI Stage 01, 02, and 03 documents are not modified by this gate.

---

## 1. Executive verdict

The previously merged dataset-evidence integration was **not accepted on CI status or prior-agent reporting alone**. The evidence chain was re-audited against the recovered source registry, early manifest, V4/V5 audit evidence, final certification evidence, model-readiness/manual-review evidence, public repository derivatives, validators, tests, and Git history.

The audit found two material provenance defects in the merged interpretation:

1. **S1 — incorrect historical complement identity.** The prior documentation stated that the 26,018-row early-manifest-to-V4 complement was the remaining PlantCity holdout plus one `rice_neck_blast` row. Machine-readable source counts instead reconcile exactly to **26,017 PlantCity rows + 1 Bangladesh row**. `rice_neck_blast` is a separate V5 under-supported-class quarantine. The separate 117,562→117,546 upstream-registry attrition is **16 rice rows**.
2. **S1 — complement provenance confidence overstated.** The 24,961 final PlantCity rows without exact historical `src_path` were labeled `verified_exact_source_family`. They are not direct per-image joins. Their source family is recovered by unique-complement reconstruction within the recovered source universe.

Both S1 defects are repaired in this gate without rewriting historical/restricted evidence bytes. The public evidence model now distinguishes direct manifest provenance from complement-derived reconstruction.

The audit also found and repaired S2 weaknesses in cross-file validation, ontology-transition discoverability, fingerprint-verification wording, external-validation exclusions, stale limitations language, and restricted-evidence ignore rules.

No unresolved S0 or S1 defect remains after the repair set described below.

---

## 2. Repository state verified

The pre-gate integration was verified on `main` at merge commit:

`fa7b44af6597cae76510a9820a660a06c2ca2897`

The three reported logical commits are present in the merged ancestry:

- `a20f51ec8b1ec7424bfff10b7530c7950bf7f5e4` — dataset-lineage evidence
- `395d63940310189adfae78368a975eeb60278de4` — dataset documentation
- `f6917062728e804294673b5d0e284a30f05e5157` — dataset contract/tests

PR #5 was merged into `main`. Its changed-file set is confined to dataset evidence, documentation, validation, and tests. No EAAI Stage-01/02/03 artifact was changed.

This QA/repair pass uses the dedicated branch:

`dataset-evidence-qa-gate-fixes`

The branch is based on the verified merged `main` state rather than on a stale pre-merge branch.

---

## 3. Lineage verification

The evidence-supported lineage is:

1. 15-source historical ingestion registry.
2. Explicit historical raw-folder label mapping and cleanup.
3. Recovered early manifest with source/path metadata.
4. V4 preflight audit over 117,546 images / 121 candidate labels.
5. V5 duplicate-route correction and leakage-controlled rebuild.
6. 109,150-image group-safe pre-manual benchmark.
7. model-readiness QA and 180-image manual-review queue.
8. 43 delete-only final exclusions.
9. certified 109,107-image / 120-class frozen benchmark.

The intermediate balancing notebook remains **historical experimental/superseded or unproven as a final ancestor** because no surviving executed fingerprint binds its output to V5. It is not used to manufacture lineage continuity.

Historical evidence is preserved as history; corrected interpretations are recorded separately.

---

## 4. Benchmark accounting

Canonical accounting:

| Transition | Rows |
| --- | ---: |
| V4 audited candidate universe | 117,546 |
| direct-cover duplicate removals | −8,355 |
| cross-label duplicate-family quarantine | −34 |
| hard-quality quarantine | −6 |
| under-supported `rice_neck_blast` quarantine | −1 |
| V5 group-safe pre-manual benchmark | **109,150** |
| final manual-review deletions | −43 |
| final frozen benchmark | **109,107** |

Final identity:

- images: **109,107**
- classes: **120**
- train: **76,376**
- validation: **16,368**
- test: **16,363**
- unique final SHA-256 identities: **109,107**
- final audited leakage-group crossings: **0**
- manual deletions: **43**
- relabels: **0**
- resplitting during final freeze: **0**

The final-freeze transition is removal-only.

---

## 5. Duplicate-edge verification

Canonical terminology is locked as follows:

| Term | Count | State | Exact meaning | Allowed manuscript wording |
| --- | ---: | --- | --- | --- |
| exact SHA-256 edges | 6,196 | V4 and V5 | exact byte-identity duplicate relationships | “6,196 exact duplicate edges” |
| strong-hash near-duplicate edges | 445 | V4 and V5 | retained strong-hash route | “445 strong-hash near-duplicate edges” |
| historical feature-route edges | 2,031 | V4 | feature-route relationships before corrected re-verification | “2,031 historical feature-route edges were re-opened” |
| retained corrected-feature edges | 1,932 | V5 | feature-route edges retained after corrected verification | “1,932 corrected feature-route edges were retained” |
| rejected feature-route edges | 99 | V5 correction | historical feature-route edges rejected after correction | “99 historical feature-route edges were rejected” |
| historical V4 confirmed set | **8,672** | historical | 6,196 + 445 + 2,031 | “historical V4 confirmed-edge set” |
| corrected V5 trusted graph | **8,573** | current | 6,196 + 445 + 1,932 | “corrected V5 trusted graph” |
| confirmed cross-label edges | 17 | V5 subset | subset of corrected trusted graph | “17 cross-label edges within the corrected trusted graph” |

The 17 cross-label edges are **not additive** to 8,573.

No current manuscript-facing evidence should call 8,672 the corrected V5 trusted graph.

---

## 6. Provenance reconstruction validation

Final provenance coverage is now expressed in two evidence classes:

### Direct recovery

**84,146 / 109,107 rows (77.1%)**

Method: `verified_manifest_join`  
Confidence: `verified_exact`

These rows are deterministically joined to the recovered historical manifest and retain exact historical `src_path` plus source-family metadata.

### Complement reconstruction

**24,961 / 109,107 rows (22.9%)**

Method: `registry_complement_reconstruction`  
Confidence: `reconstructed_unique_complement`

These rows have no surviving exact historical `src_path`. Their source family is reconstructed as PlantCity Pakistan by exhaustive recovered-registry/final-accounting complement.

This confidence is conditional on:

- completeness of the recovered source universe for the audited lineage;
- no untracked intermediate source-family introduction;
- unambiguous accounting of every other final source family.

Therefore the 24,961 rows support **source-family attribution within the recovered source universe**, but not direct source-row identity, original path recovery, image-level licensing, or external-independence claims.

The previous public confidence phrase `verified_exact_source_family` is formally superseded. The restricted 109,107-row ledger is not rewritten.

---

## 7. Source registry validation

The historical registry preserves **15 registered source families**. Ten have surviving final rows.

Final source composition:

| Source family | Final rows |
| --- | ---: |
| plantvillage | 39,973 |
| plantcity_pk | 33,872 |
| sugarcane | 13,671 |
| plant100k | 7,903 |
| abcgmp | 6,833 |
| bangladesh | 4,092 |
| plantdoc | 2,559 |
| coleaf | 119 |
| corn | 67 |
| rice | 18 |
| **Total** | **109,107** |

The repaired validator reconciles each final source independently across:

1. historical source registry final count;
2. `final_source_counts.csv`;
3. sum of source-by-split counts;
4. sum of provenance-coverage rows.

A mismatch in any one source now fails the evidence contract.

### Historical count reconciliation

Recovered upstream registry: **117,562**  
V4 audited universe: **117,546**

Difference: **16 rows**, entirely in the rice family:

`34 recovered rice registry rows − 18 V4 rice rows = 16`

Recovered early manifest: **91,528**  
V4 audited universe: **117,546**

Difference: **26,018 rows**, exactly:

- PlantCity Pakistan: **26,017**
- Bangladesh: **1**
- all other source families: **0**

This corrects the earlier `rice_neck_blast` complement error.

---

## 8. PlantCity holdout analysis

PlantCity was historically configured with holdout intent.

That historical intent does not survive as final external-validation independence because later V5 reconstruction treats the audited universe as provenance input and creates a new leakage-group-safe split.

Formal lock:

> **Historical holdout intent ≠ final independent external validation.**

PlantCity contributes **33,872 final internal-benchmark rows**. It must not be described as a final independent Pakistan test cohort.

All 15 historical ingestion sources are now included in a machine-readable external-evaluation exclusion registry. Any future external cohort derived from one of these source families remains excluded unless new source-family and image-overlap evidence establishes independence.

---

## 9. Ontology validation

The ontology is a harmonized **plant-health recognition vocabulary**, not “120 diseases.”

Verified historical states include:

- 249 explicit raw-folder mapping rows;
- 157 possible historical mapped target strings;
- historical case-sensitive mappings and heuristics;
- historical 80-class exported `class_index.json` state;
- 121-label V4 audited candidate ontology;
- 120-class final operational ontology.

Explicit historical transformations now have a machine-readable transition registry:

- case normalization / cleanup;
- `tomato_leaf_bacterial_spot → tomato_bacterial_spot`;
- collapse of 17 nutrient-deficiency variants into `nutrient_deficiency`;
- historical support filtering / intermediate exported state;
- 121-class V4 candidate state;
- `rice_neck_blast` quarantine producing the final 120 classes.

Historical case-sensitive ambiguities are explicitly disclosed, including:

- `Smut → wheat_smut` versus `smut → wheat_loose_smut`;
- `healthy → cotton_healthy` versus `Healthy → __NEEDS_CROP_PREFIX__`.

These are historical construction facts, not silently normalized away in the evidence record.

---

## 10. QA/manual-review validation

Recovered model-readiness evidence supports:

- residual cross-label embedding-similarity pairs: **122**
- unique pair endpoints: **180**
- supplied manual-review priority queue: **180**
- pair-endpoint set equals review-queue set: **true**

The observed priority scores are exactly reproduced by:

`4 + 3*embedding_review_flag + soft_label_review_flag + (cleanlab_flag AND low_given_label_probability) + flag_visual_label_outlier_v5`

The repository does **not** call this the original historical generator implementation because that source code is not present.

Final manual-review evidence supports exactly **43 deletions**:

- train: 29
- validation: 8
- test: 6
- relabels: 0
- final resplits: 0

All 43 are delete actions for severe black-screen/image-distortion findings.

---

## 11. Fingerprint/certificate verification

Recorded final identities:

- final manifest fingerprint: `7c368e6e3d8be3bb3a9a3a5f961075d4faa125bcac2e98a3b55e1a1c61f1c523`
- final build fingerprint: `2d7c237981b8943d9b08a522db0489849a4bfe7f488dfb6461489dd36dcdc12b`
- restricted provenance ledger SHA-256: `fe596ead93eff17a5b976b69e347ef614dd38e6d90efd41ffa0d128c54fc7181`

A prior validator weakness was confirmed: public checks compared repeated constants and did not independently reconstruct the restricted final-manifest/build fingerprints.

The repaired evidence model now states this explicitly:

- final manifest fingerprint: **public cross-file consistency only**
- final build fingerprint: **public cross-file consistency only**
- independent public recomputation: **false**
- authorized recomputation: requires the restricted canonical bytes/build evidence

The validator now fails if public documentation/data falsely claims independent recomputation.

This limitation does not invalidate the certificate; it bounds what the public repository alone can prove.

---

## 12. Evidence-registry audit

The evidence registry preserves:

- unique evidence IDs;
- stage;
- artifact name;
- role/description;
- historical/current status;
- SHA-256 identity;
- file size;
- row count where applicable;
- schema summary;
- visibility;
- supported claims;
- public derivative where applicable.

The repaired validator checks:

- evidence-ID uniqueness;
- SHA-256 field shape;
- required schema;
- existence of every declared public derivative;
- explicit interpretation-supersession status for the restricted provenance ledger.

The restricted ledger remains identified by its original immutable hash. Its confidence interpretation is superseded externally rather than by changing the artifact bytes.

---

## 13. Validator/test audit

### Before this gate

The validator had useful coverage but several checks were only partially independent:

| Area | Prior quality |
| --- | --- |
| final counts | partially independent |
| certificate fingerprints | largely consistency/constant checks |
| duplicate-edge arithmetic | partially independent |
| provenance total | partially independent |
| provenance per-source reconciliation | missing |
| historical complement identity | missing |
| confidence semantics | missing |
| external-contributor exclusion | missing |
| ontology transition details | missing |
| restricted path/secret scan | genuinely useful |
| document presence | presence-only |

### After repair

The validator now independently cross-reconciles:

- recovered registry total;
- recovered early-manifest total;
- V4 audited total;
- 16-row registry→V4 rice attrition;
- 26,018-row early-manifest→V4 complement;
- exact **26,017 PlantCity + 1 Bangladesh** breakdown;
- final per-source counts across four public derivatives;
- provenance confidence semantics;
- external-exclusion coverage for all 15 historical sources;
- ontology-transition locks;
- final benchmark counts and zero crossing claims;
- duplicate-edge terminology;
- final review accounting;
- evidence-registry derivative existence;
- restricted-path and secret guards;
- stale dangerous wording;
- public fingerprint-verification scope.

Unit tests mirror the highest-risk scientific contracts instead of checking only file presence.

---

## 14. CI audit

The previous integration was verified to have successful PR-head and merged-main GitHub Actions runs executing:

1. `python scripts/validate_repository.py --strict`
2. `python -m unittest discover -s tests -v`

The workflow has no `continue-on-error`, no path filter that bypasses evidence changes, and no skipped test command.

Current repair validation:

- PR: **#6 — Close dataset-evidence QA gate for EAAI amendments**
- validated head: `8cf3d43eafcb39cf297dad24e59f26421bff0bbf`
- GitHub Actions run: **33893233205**
- strict repository validator: **PASS**
- unit-test suite: **PASS**

The final gate commit is revalidated once more after this status is written. The merged `main` tree is also checked after merge before Stage-01/02/03 amendment work begins.

---

## 15. Privacy/security audit

The public repository boundary excludes:

- source image corpus;
- raw audit ZIPs;
- large row-level forensic tables;
- model checkpoints;
- PTE/model binaries;
- credentials/tokens;
- private runtime paths.

The 43,837,322-byte reconstructed row-level provenance ledger remains restricted under SHA-256:

`fe596ead93eff17a5b976b69e347ef614dd38e6d90efd41ffa0d128c54fc7181`

The repository validator scans text-like public files for common secret patterns, private local paths, and Kaggle runtime paths in public dataset-evidence directories, rejects forbidden model/binary suffixes, and rejects files above the public size threshold.

The ignore rules were strengthened so nested restricted directories may retain a public `README.md` boundary document without opening the rest of the restricted tree to accidental commits.

---

## 16. Contradictions found

| Severity | Defect | Resolution |
| --- | --- | --- |
| S1 | V4 historical-test complement incorrectly described as PlantCity + one `rice_neck_blast` row | corrected to **26,017 PlantCity + 1 Bangladesh**; rice attrition/quarantine separated |
| S1 | 24,961 complement-derived PlantCity rows labeled `verified_exact_source_family` | superseded by `reconstructed_unique_complement`; restricted bytes preserved |
| S2 | final source totals tested mainly as grand totals | validator now reconciles each source across registry/count/split/provenance tables |
| S2 | fingerprint checks could be read as independent recomputation | explicit fingerprint-verification-status registry added; public scope locked to cross-file consistency |
| S2 | ontology history lacked transition-level detail and case-sensitive ambiguities | machine-readable ontology transition registry added |
| S2 | no machine-readable historical-source external-evaluation exclusion lock | 15-source exclusion registry added |
| S2 | limitations said source-level provenance was incomplete | wording corrected to distinguish source-family attribution from path/licence incompleteness |
| S3 | nested restricted README ignore behavior was fragile | `.gitignore` hardened |

No S0 defect was identified.

---

## 17. Enhancements performed

Each enhancement addresses a concrete failure mode:

1. **Historical complement registry** — prevents PlantCity/Bangladesh/rice chronology from drifting in prose.
2. **Provenance-confidence definitions** — prevents complement inference from being presented as direct per-image provenance.
3. **External-evaluation exclusion registry** — prevents a historical contributor or repackaging from being selected later as a supposedly independent external cohort.
4. **Ontology-transition registry** — prevents silent merges/collapses and the “120 diseases” mischaracterization.
5. **Fingerprint-verification-status registry** — prevents hard-coded consistency checks from being presented as cryptographic recomputation.
6. **Per-source cross-file validator** — prevents grand totals from masking one-source discrepancies.
7. **Stale-wording assertions** — prevents the known dangerous complement/provenance phrases from returning unnoticed.
8. **Nested restricted-evidence ignore hardening** — preserves discoverable boundary documentation without exposing restricted artifacts.

No cosmetic restructuring was performed.

---

## 18. Files/commits changed by this gate

### New canonical files

- `data_card/provenance/historical_test_complement.csv`
- `data_card/provenance/provenance_confidence_definitions.json`
- `data_card/provenance/external_evaluation_exclusion_registry.csv`
- `data_card/lineage/ontology_transition_registry.csv`
- `data_card/audit/fingerprint_verification_status.json`
- `00_EAAI_DATASET_EVIDENCE_QA_GATE.md`

### Corrected/strengthened files

- `data_card/provenance/provenance_coverage.csv`
- `data_card/lineage/lineage.json`
- `docs/DATASET_LINEAGE.md`
- `docs/DATASET_PROVENANCE.md`
- `docs/PROVENANCE_STATUS.md`
- `docs/EVIDENCE_BOUNDARIES.md`
- `docs/LIMITATIONS.md`
- `data_card/README.md`
- `README.md`
- `evidence/restricted/dataset_lineage/README.md`
- `evidence/public/dataset_lineage/evidence_registry.csv`
- `evidence/public/claim_evidence_matrix.csv`
- `scripts/lib/repo_contract.py`
- `scripts/validate_repository.py`
- `tests/test_repository_contract.py`
- `.gitignore`

Logical repair commits:

- `13a5be822ee9ebf2ef900619369c9127a299610b` — data: correct provenance evidence and add Q1 locks
- `cf35121ea632cdebab9504f91286be8f1c678d60` — docs: repair dataset provenance interpretation
- `a4d37491910217698c7ff760c79c89402730d144` — test: harden dataset evidence QA contract
- `8cf3d43eafcb39cf297dad24e59f26421bff0bbf` — docs: add independent EAAI dataset evidence gate

PR: **#6**.

---

## 19. Remaining limitations

The following limitations remain real and should be inherited by the journal extension rather than “fixed” by inference:

1. exact original historical `src_path` is unrecoverable for 24,961 final PlantCity-attributed rows;
2. complement-derived attribution is conditional on completeness of the recovered source universe and absence of an untracked source-family introduction;
3. image-by-image redistribution rights are not established by source-family reconstruction;
4. the historical model-readiness priority-generator source code does not survive;
5. the intermediate balancing notebook is not proven to be a final-lineage ancestor;
6. the exact historical reason for the 16 rice registry rows disappearing before V4 is not recovered;
7. public Git cannot independently recompute the final restricted-manifest/build fingerprints from canonical bytes;
8. internal group-safe evaluation is not source-independent field validation.

These limitations are disclosed and do not alter the frozen final benchmark identity.

---

## 20. Inputs for the Stage-01/02/03 Amendment Pass

The superseding Stage-01/02/03 amendment pass must inherit all of the following as hard inputs:

1. **Dataset identity:** 109,107 images; 120 operational plant-health classes; 76,376 / 16,368 / 16,363 train/validation/test.
2. **V5 pre-manual state:** 109,150 images; final freeze is deletion-only by exactly 43 rows.
3. **Duplicate terminology:** 8,672 = historical V4 confirmed set; 8,573 = corrected V5 trusted graph; 17 cross-label edges are a subset of 8,573.
4. **Historical leakage finding:** 3,233 historical cross-split duplicate relationships; final audited leakage-group crossings = 0.
5. **Source registry:** 15 historically registered families; ten contribute final rows.
6. **Source composition:** use the frozen final per-source counts in `final_source_counts.csv`.
7. **Provenance:** 84,146 direct manifest joins (`verified_exact`); 24,961 PlantCity source-family reconstructions (`reconstructed_unique_complement`), not direct path recovery.
8. **Historical complement correction:** 91,528→117,546 delta = 26,018 = 26,017 PlantCity + 1 Bangladesh.
9. **Separate rice fact:** 117,562→117,546 upstream-registry attrition = 16 rice rows; do not conflate this with `rice_neck_blast`.
10. **PlantCity:** historical holdout intent does not constitute final external independence.
11. **External validation:** every historical ingestion source is excluded as a primary independent external cohort unless a new cohort passes independence/overlap checks.
12. **Ontology:** describe 120 operational **plant-health classes**, not 120 diseases; inherit the explicit ontology-transition registry.
13. **Manual review:** 43 severe-quality deletions; 0 relabels; 0 final resplits.
14. **Model-readiness queue:** 122 residual cross-label pairs → 180 unique endpoints = 180 review rows; observed score formula is reproducible, original generator source is not preserved.
15. **Fingerprints:** cite the final manifest/build fingerprints as certified identities; do not claim public independent recomputation.
16. **Evidence scope:** restricted provenance/manifests remain valid evidence even though they are not publicly redistributed.
17. **External claims:** no source-independent field generalization, physical Android performance, or image-rights claim is established by the current dataset foundation.
18. **Planning action:** Stage 01/02/03 should be produced as clean superseding revisions, not repeated ad-hoc patches to stale assumptions.

---

## 21. Formal gate

The independent audit identified the material provenance discrepancies, repaired them without rewriting historical/restricted evidence, strengthened the evidence contract to detect their recurrence, and obtained a passing strict validator and unit-test run on the repair PR head.

There are **no unresolved S0 defects and no unresolved S1 defects**. Remaining limitations are explicitly bounded in Section 19 and do not change the frozen benchmark identity.

The repository is scientifically and structurally suitable to become the authoritative dataset-evidence input for the superseding EAAI Stage-01/02/03 amendment pass, subject to the operational requirement that the identical validated tree is merged to `main` and the merged-main CI remains green.

## `GO — DATASET EVIDENCE FOUNDATION VERIFIED`
