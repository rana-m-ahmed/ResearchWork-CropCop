# Evidence boundaries

## Established

- 109,107-image, 120-class frozen internal benchmark.
- No certified leakage-group crossings across final partitions.
- Historical V4 duplicate set (8,672) is separated from the corrected V5 trusted graph (8,573); the 17 cross-label edges are a subset of the latter.
- Source-family attribution for all 109,107 final rows; exact original-source-path recovery for 84,146 rows.
- Delete-only 43-image final manual-review transition with no relabeling or resplitting.
- Locked-test reference, compact-float, converted-INT8, and direct-PTE metrics.
- Exact artifact hashes, PTE byte size, PTQ selection record, and six-case serialisation disagreement audit.

## Suggestive but not causally isolated

- The benefit of DINOv3 pretraining relative to a matched control.
- The benefit of teacher-guided training relative to direct MobileNetV4 training.
- Mechanisms behind class-level retention and loss.

## Not established

- Source-independent field or smartphone generalisation.
- PlantCity as an independent final external holdout; it contributes to the pooled/group-safe internal benchmark.
- Exact original per-image source path for 24,961 final PlantCity-attributed rows.
- Image-by-image redistribution rights.
- Multi-seed training stability.
- Physical Android latency, memory, energy, or thermal behaviour.
- Safety for autonomous agronomic diagnosis or treatment decisions.
