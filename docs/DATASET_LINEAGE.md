# CropCop dataset lineage

This document is the canonical chronology for the dataset evidence recovered for the EAAI journal extension. Primary uploaded audit/manifests take precedence over earlier notebook-forensic orientation. Historical artifacts are preserved as historical evidence rather than silently corrected.

## Authoritative chronology

### A. Upstream public-source aggregation

The historical source registry names **15 registered ingestion families**. Ten have rows in the recovered registry and contribute to the audited/final lineage: PlantVillage, PlantDoc, ABCGMP, CoLeaf, Plant100K, PlantCity Pakistan, Bangladesh, rice, corn, and sugarcane. Five registered families have no surviving rows in the recovered registry and are therefore not asserted as final contributors.

`final_cropcop_registry.csv` contains **117,562** recovered registry rows. The V4 audited universe contains **117,546**. The 16-row difference is confined to the historical `rice` registry family: the registry contains 34 rice rows, while 18 occur in the V4 audited universe.

### B. Ontology harmonization

The recovered historical label-map ledger contains **249 raw-folder mappings** to **157 possible target strings**. It is historical notebook evidence, not proof that every target was instantiated. An uploaded historical `class_index.json` contains **80 classes** and is preserved as an earlier state, not the final class-map authority.

The V4 audited candidate universe contains **121 labels**. V5 quarantines the single under-supported `rice_neck_blast` row, producing the **120-class operational ontology** used by the frozen benchmark.

### C. Early cleaning and historical split

The recovered `dataset_manifest.csv` contains **91,528 rows** with `src_path`, merged destination path, source-family key, domain, weight metadata, and historical split. Its recorded split is 57,426 train, 12,272 validation, and 21,830 test rows.

The V4 audit later sees the same 57,426 train and 12,272 validation rows but **47,848 historical test rows**. The recovered early manifest-to-V4 difference is exactly **26,018 rows**. Source-count reconciliation resolves that delta as **26,017 additional PlantCity Pakistan rows plus one Bangladesh row**. The one-image `rice_neck_blast` category is a separate V5 under-supported-class quarantine; it is **not** the non-PlantCity row in this historical test complement. The separate upstream-registry-to-V4 discrepancy is 16 rice rows (34 registry rows versus 18 V4 rows), i.e. pre-V4 attrition rather than test-complement growth.

### D. Intermediate balancing/rebuild branch

The historical balancing notebook defines a separate re-dedup/re-split/augmentation workflow, but the recovered evidence does not contain an executed output fingerprint that ties that branch to the final V5 lineage. It is therefore classified as **historical experimental/superseded, not proven final ancestor**.

### E. V4 preflight audit

V4 audits **117,546 images** over **121 labels**. It reports 8,672 historical confirmed duplicate relationships, 9,721 borderline pairs, 34 cross-label conflict images, and 20 ambiguous duplicate chains. The precheck fails on label-conflicting duplicate families and ambiguous chains.

The 8,672 V4 set is composed of 6,196 exact-SHA edges and 2,476 near-duplicate edges. The near-duplicate edges split into 445 strong-hash-route edges and 2,031 feature-route edges.

### F. V5 repair

V5 reopens all **2,031 feature-route edges** because the earlier ORB/geometric route required corrected verification. The corrected pass retains **1,932** and rejects **99**. Therefore the corrected trusted graph is:

`6,196 exact + 445 strong-hash + 1,932 corrected-feature = 8,573 trusted edges`.

The **17 confirmed cross-label edges are a subset of those 8,573**, not an additive set.

### G. V5 benchmark reconstruction

V5 removes **8,355** direct-cover duplicates and quarantines **41** rows: 34 cross-label duplicate-family images, six hard-quality failures, and one under-supported `rice_neck_blast` row. The resulting group-safe benchmark contains **109,150 images / 120 classes**: 76,405 train, 16,376 validation, and 16,369 test.

The V5 manifest has 109,150 unique SHA-256 values and 109,133 leakage groups. Certification reports **zero leakage-group crossings** and **zero trusted-edge crossings** across final partitions. Five-fold CV assignments are also group-safe.

### H. Model-readiness QA

The recovered QA bundle is `1.3.0-final-robust`, status `PASS_WITH_QUARANTINE`, with **zero critical failures** and seven warning-level findings. It retains the V5 manifest identity.

The 180-row manual-review queue is reconstructable from the recovered outputs: the **122 residual cross-label embedding-similarity pairs have exactly 180 unique endpoints**, and that endpoint set is exactly the 180-row review queue. The queue's priority score also reproduces for all 180 rows from the supplied flags. The exact historical generator source code is not present, so this establishes queue membership and scoring semantics, not implementation identity.

### I. Manual review and final freeze

Manual review deletes **43 images**, all for `Black-screen / severe image distortion identified during manual visual inspection`. There are **no relabels and no resplitting operations**. Deletions are 29 train, 8 validation, and 6 test rows.

The transition is exactly **109,150 → 109,107**. The certified final benchmark contains **76,376 train / 16,368 validation / 16,363 test**, 120 classes, 109,107 unique SHA-256 values, zero leakage-group crossings, and a removal-only finalization certificate.

## Final identity

- final manifest fingerprint: `7c368e6e3d8be3bb3a9a3a5f961075d4faa125bcac2e98a3b55e1a1c61f1c523`
- final build fingerprint: `2d7c237981b8943d9b08a522db0489849a4bfe7f488dfb6461489dd36dcdc12b`
- manual-review fingerprint: `05bc92da66ca8fdcb0bfe3e1fdf3df3585f1cdc0ab429f1fafa94b81057e079b`
- source QA manifest fingerprint: `f298f5bc3a3492d6c14ae4e6c4bc60f2d78fdab924ff0ba9d6e53002747399a5`

Machine-readable chronology is in `data_card/lineage/lineage.json`.
