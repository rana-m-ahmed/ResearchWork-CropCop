# Teacher Class-Order Rationale — Historical Stage-1 DINOv3 Reference

This record reconstructs historical semantics; it does not create a new scientific definition.

The historical v7 trainer maps each manifest row label through the frozen `class_to_idx` mapping and uses the resulting integer directly as the 120-way classification target. Its DINO wrapper returns `head(dropout(pooler_output))` directly and contains no output-index permutation.

The certified final QA engine independently:

1. verifies the exact frozen class-map SHA;
2. requires class-map values to be exactly 0..119;
3. recreates targets with `df["label"].map(class_to_idx)`;
4. reconstructs the same DINO wrapper;
5. strict-loads the raw state;
6. applies complete EMA shadow coverage;
7. interprets the resulting 120 logits directly.

Therefore the strongest evidence-supported semantics are:

- `class_map_sha256 = 46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`;
- `historical_index_semantics = class_map_index`;
- `classifier_weight_key = head.weight`;
- `classifier_shape = [120,768]`;
- `output_order_transform = none`;
- `not_inferred_from_shape_only = true`.

The proof is grounded in the preserved historical source excerpts in this directory. The actual teacher checkpoint remains private and is verified by exact SHA only when mounted during the non-qualifying readiness/G1 path.
