# Evidence boundaries — EAAI journal extension

## Established

### Benchmark and selection

- 117,546-image audited candidate universe.
- 8,573 corrected trusted duplicate relations.
- 8,355 direct-cover duplicate-row removals.
- 109,107-image, 120-class frozen benchmark; 76,376 / 16,368 / 16,363 split.
- Zero audited trusted leakage-group crossings among final partitions.
- Frozen Track-A selector reproduces R07 / ConvNeXt-Tiny.
- External predictions and Track-C device outcomes are unavailable to upstream family selection.

### External-source evidence

- Prediction-blind retention of GVLiD v5 and Irish Potato v01.
- Exact seven-class mapping into the native 120-class output space.
- R07 S1/S2/S3 all preserved.
- GVLiD mean macro-F1 0.3315 ± 0.0162; OOS 0.3637.
- Irish Potato mean macro-F1 0.4160 ± 0.0326; OOS 0.6582.
- One-per-SHA sensitivity: 0.2507 and 0.3646 macro-F1 respectively.
- OOS is out-of-mapped-scope routing, not open-set recognition.
- One-per-SHA results are composition sensitivity, not uniquely “true” performance.

### Runtime lineage

- Exact `R07-CNXTT-CONTEXT-S1` checkpoint identity and epoch.
- 16,368-row validation replay within `1e-6`.
- FP32/INT8 artifacts identity-bound.
- FP32→INT8 canonical top-1 = 252/256; changes 11, 27, 155, 255.
- Pixel FP32 host-device = 256/256.
- Pixel INT8 host-device = 256/256.
- Pixel raw-vs-canonical INT8 = 254/256; mismatches 11, 255.
- POCO original runtime: load succeeds, first forward SIGILL, zero fidelity rows.
- POCO compatible runtime: canonical 255/256 and raw 253/256.

## Bounded but not causal

- Teacher-guided versus direct MobileNetV4: three paired states, descriptive only.
- R12 logits-versus-feature contrast: descriptive auxiliary evidence.
- Grad-CAM++: trustworthiness/shortcut audit, not lesion localization or biological explanation.
- Pixel FP32-versus-INT8 latency: separate APK revisions and nonstationarity; not a causal or universal speedup.
- POCO compatibility: configuration-dependent executability; not proof of one unique low-level cause.

## Not established

- Universal field/geographic/cultivar/camera generalization.
- Open-set recognition or safe unsupported-input rejection.
- Production readiness or autonomous agronomic safety.
- Android-wide runtime equivalence.
- Universal real-time behavior.
- Energy efficiency; energy was not measured.
- Pixel-versus-POCO performance ranking.
- Causal architecture-only superiority.
- Public availability of restricted bytes merely because hashes are published.

## Availability boundary

Public Git contains code, protocols, locks, aggregate/derived evidence, runtime measurements, validators and paper-facing documentation. It does not currently redistribute the consolidated image corpus, final checkpoint, produced PTE binaries, raw logits or the complete accepted private Track-B evidence archive.
