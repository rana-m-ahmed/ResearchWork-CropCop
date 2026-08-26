# CropCop manuscript status — v0.2.0-rc2

## Current status

The manuscript **“CropCop: An Auditable 120-Class Plant-Health Model from Benchmark Reconstruction to a Quantised Runtime Artifact”** has been submitted to arXiv.

- Submission state: **submitted; public arXiv identifier pending**
- Submission does not imply public announcement, endorsement, acceptance, or peer review.
- The repository will add the canonical arXiv abstract-page URL and identifier only after assignment.
- The scientific claim boundary remains unchanged while the identifier is pending.

## Authorship

- Rana Muhammad Ahmed — corresponding author
- Sabahat Abbas — co-author
- Department of Computer Science, Bahria University Islamabad, Islamabad, Pakistan

## Submitted manuscript identity

The submitted paper reports the final 120-class audit-to-runtime study, including:

- 117,546 audited source images;
- a 109,107-image frozen benchmark;
- 3,233 confirmed historical cross-split duplicate relationships;
- zero crossings among the audited trusted leakage groups in the final split;
- a DINOv3 ConvNeXt-Tiny reference;
- a compact MobileNetV4 Conv-Medium lineage;
- validation-only post-training quantisation selection;
- direct execution of the final 22.60 MiB ExecuTorch/XNNPACK PTE;
- row-level and paired analysis of the final runtime state.

The public repository identifies the evaluated dataset, model states, and runtime artifact through fingerprints and cryptographic hashes. Restricted binaries and source data remain outside public Git history.

## Completed editorial and integrity checks

- Both authors are listed consistently in manuscript and repository citation metadata.
- The manuscript links to the exact companion repository URL.
- Human-facing metadata uses the capitalization **arXiv** consistently.
- The paper distinguishes software-runtime execution from physical Android evidence.
- Host latency claims are excluded where CPU, thread-count, and operating-system details were not archived.
- Internal recognition results are not presented as field generalisation.
- The compact-model result is not presented as causal evidence for a new distillation method.
- Dataset, model, and runtime identities are bound to public registries and checksums.

## Repository release boundary

The current repository metadata version is `0.2.0-rc2`.

This branch publishes the public evidence bootstrap, metric registry, claim ledger, model/data identity records, validation utilities, and submission-aware documentation. It does not publish:

- raw source images;
- model checkpoints;
- the final PTE binary;
- raw logits or large prediction bundles;
- credentials;
- restricted forensic evidence.

The complete submitted source package and compiled paper should be synchronized only through a reviewed release process that preserves the exact submitted version and excludes restricted material.

## Next release action

After arXiv assigns the identifier:

1. verify the public abstract-page metadata against the submitted title and author order;
2. update `README.md`, `CITATION.cff`, and `CITATION.bib` with the canonical identifier and URL;
3. bind the repository release manifest to the submitted PDF/source package and checksums;
4. create an immutable preprint tag only after repository validation passes;
5. avoid changing scientific results under the same release identity.
