# Releases

Large or redundant research assets belong in GitHub Releases rather than ordinary Git history.

## Current state

The CropCop manuscript has been submitted to arXiv. The public identifier is pending, so no immutable arXiv-linked release should be created yet.

## Planned preprint release

After the arXiv identifier is assigned and verified, the planned immutable preprint release should contain only redistributable artifacts:

- canonical submitted PDF;
- arXiv source archive;
- public evidence archive;
- release manifest;
- checksum ledger;
- citation metadata carrying the assigned arXiv identifier;
- repository validation report.

The release must not include raw source images, restricted forensic bundles, credentials, model checkpoints, raw logits, or the final PTE binary unless redistribution obligations have been reviewed and explicitly cleared.

## Release gate

Before tagging:

1. verify title, author order, and arXiv identifier;
2. confirm the PDF/source checksums against the submitted package;
3. run `python scripts/validate_repository.py --strict`;
4. run the repository test suite;
5. inspect the public archive for restricted paths and binaries;
6. update `README.md`, `CITATION.cff`, and `CITATION.bib`;
7. create the tag only after all checks pass.

A new scientific result or changed manuscript must receive a new version identity rather than silently replacing an existing release.
