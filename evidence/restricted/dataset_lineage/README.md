# Dataset-lineage restricted evidence boundary

The public repository intentionally excludes the raw audit archives, row-level manifests containing working paths, image corpora, contact sheets, and large forensic tables. Their identities are registered in `evidence/public/dataset_lineage/evidence_registry.csv`.

The journal-extension provenance reconstruction also produced a restricted row-level ledger:

- logical name: `source_provenance_reconstruction_restricted.csv`
- rows: **109,107**
- bytes: **43,837,322**
- SHA-256: `fe596ead93eff17a5b976b69e347ef614dd38e6d90efd41ffa0d128c54fc7181`
- role: bridge final `record_key`/SHA/split/leakage group back to the recovered ingestion source family and, when available, exact original source path.

It is not committed because it contains original working/source paths and is larger than the public-evidence threshold. Public aggregate derivatives preserve its scientific conclusions without exposing those paths.

## Provenance-confidence supersession note

The restricted 109,107-row reconstruction is preserved byte-for-byte under the SHA-256 above. If its complement-derived PlantCity rows carry the historical confidence label `verified_exact_source_family`, that wording is **superseded in the public evidence model** by `reconstructed_unique_complement`. The supersession changes interpretation, not the restricted artifact bytes. See `data_card/provenance/provenance_confidence_definitions.json`.
