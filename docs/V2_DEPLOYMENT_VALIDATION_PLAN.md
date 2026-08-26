# CropCop V2 Deployment and External Validation Plan

## Purpose

Version 1 establishes leakage-controlled internal recognition and direct software-runtime fidelity for one identified ExecuTorch/XNNPACK PTE. Version 2 is intended to extend the evidence boundary without reusing the consumed internal test set for model, threshold, or quantisation selection.

This document is a planning specification, not evidence that the listed experiments have been completed.

## Evidence tracks

V2 should separate two distinct questions:

1. **Physical deployment:** does the frozen PTE execute correctly and efficiently on representative Android hardware?
2. **External validity:** how well does the frozen system perform on independently collected smartphone images outside the reconstructed source pool?

Device benchmarks alone do not establish field generalisation. External-image accuracy alone does not establish on-device runtime behavior.

## Track A — Android deployment validation

### Frozen inputs

Before device testing, freeze and record:

- application commit;
- PTE SHA-256 and byte count;
- class-map SHA-256;
- preprocessing implementation and configuration;
- ExecuTorch, PyTorch, Android, NDK, and build-tool versions;
- thread count and affinity settings;
- backend and delegate configuration;
- test-image manifest and expected outputs.

### Device matrix

Use at least three device tiers where feasible:

- low-resource Android device;
- representative mid-range device;
- modern upper-mid or flagship reference device.

For every device record manufacturer, model, SoC, CPU architecture, RAM, Android version, battery state, thermal state, and power mode.

### Measurements

Measure and report distributions rather than a single average:

- cold application start;
- model loading and initialization;
- preprocessing latency;
- model inference latency;
- postprocessing latency;
- end-to-end image-to-result latency;
- median, p90, p95, minimum, and maximum latency;
- peak resident memory and incremental inference memory;
- application package and installed footprint;
- sustained performance over at least 100 consecutive inferences;
- thermal change and latency drift;
- crashes, delegate fallbacks, unsupported operators, and numerical anomalies;
- energy per inference where defensible measurement is available.

### Runtime fidelity

For a fixed test manifest, compare device outputs with the archived software-runtime outputs using:

- top-1 agreement;
- top-k agreement;
- maximum absolute and relative logit difference where logits are exposed;
- changed-decision audit;
- confidence and margin changes;
- preprocessing parity checks.

Any platform-specific difference must be reported as a new evaluated state rather than inheriting the archived PTE metrics.

## Track B — Source-independent smartphone cohort

### Cohort design

The external cohort should be collected after V1 model and threshold choices are frozen. The protocol should define:

- supported crops and conditions;
- inclusion and exclusion criteria;
- geographic and seasonal scope;
- capture devices and acquisition conditions;
- image-count targets by class;
- label-verification process;
- mapping from field labels to the 120-class operational ontology;
- handling of ambiguous, mixed, or unsupported cases.

### Test integrity

- Do not use the external test cohort for training or hyperparameter selection.
- Use a separate development set if field-specific calibration or thresholds are explored.
- Preserve stable row identifiers and a frozen manifest.
- Record exclusions and reasons before final evaluation.
- Report support for every class and subgroup.

### Evaluation

At minimum report:

- accuracy;
- balanced accuracy;
- macro-F1 and weighted F1;
- per-class precision, recall, F1, and support;
- top-3 and top-5 accuracy;
- calibration metrics;
- confidence intervals;
- risk-coverage curves;
- results by device, lighting, background, crop organ, severity, and image quality where support permits.

## Track C — Unsupported-input and abstention evaluation

Because CropCop is a closed-set classifier, V2 should test inputs outside the supported ontology:

- unsupported crops;
- novel or unmapped diseases;
- non-plant images;
- severely blurred or obstructed images;
- multiple plants or mixed symptoms;
- screenshots and synthetic transformations.

Evaluate an abstention or review mechanism using prespecified metrics such as:

- AUROC and AUPR for supported versus unsupported inputs;
- false-positive rate at a fixed supported-input recall;
- accepted-set accuracy and coverage;
- false-acceptance analysis;
- subgroup-specific failure examples.

No internal validation threshold should be presented as a field policy without external validation.

## Track D — Controlled compact-model analysis

A scientifically stronger V2 may add a new, fully specified compact-model experiment:

- matched direct MobileNetV4 baseline;
- matched teacher-guided MobileNetV4 run;
- identical data, initialization policy, augmentations, optimizer family, schedule, and evaluation protocol;
- multiple seeds where compute permits;
- validation-only selection;
- no selection using the consumed V1 test set.

This new experiment must receive a new configuration and artifact identity. It must not be presented as reconstruction of missing historical V1 optimizer or objective settings.

## Required release artifacts

A complete V2 evidence package should include:

- prespecified protocols;
- device and software environment registers;
- external-cohort manifest and provenance statement;
- raw or minimally processed measurement tables where redistribution is allowed;
- deterministic analysis scripts;
- paired prediction records;
- model, data, application, and artifact fingerprints;
- claim-to-evidence matrix;
- limitations and intended-use updates;
- versioned checksums and release manifest.

## Decision gates

### G1 — Identity

All model, app, class-map, preprocessing, and evaluation identities are frozen and recorded.

### G2 — Device correctness

On-device outputs are evaluated directly and any discrepancies are auditable.

### G3 — Measurement quality

Latency, memory, thermal, and stability results use documented procedures and representative repeated measurements.

### G4 — External integrity

The independent cohort is not used for model or threshold selection and has a prespecified ontology mapping.

### G5 — Safety boundary

Unsupported inputs, abstention behavior, and human-review requirements are explicitly evaluated and documented.

### G6 — Release integrity

The manuscript, repository, evidence package, citation metadata, and checksums refer to the same versioned result.

## Non-claims until completion

Until these tracks are completed, the repository must not claim:

- validated field diagnosis;
- autonomous agronomic decision support;
- universal mobile performance;
- complete XNNPACK operator delegation;
- robustness to unseen crops or diseases;
- causal benefit from teacher guidance;
- production readiness.
