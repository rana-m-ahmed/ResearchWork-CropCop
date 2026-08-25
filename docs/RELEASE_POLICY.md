# Release policy

- `v0.1.0-bootstrap`: repository structure, paper, and public evidence before final figures/arXiv ID.
- `v1.0-preprint`: final figures, author metadata, arXiv source, PDF, public evidence bundle, and checksums.
- `v1.0.1-correction`: non-substantive corrections with a correction note.
- `v2.0-field-evaluation`: future external smartphone/device study; v1 evidence remains immutable.

A release is built from a frozen commit. Release assets must match `release_manifest.json`; moving `main` is not a reproducibility reference.
