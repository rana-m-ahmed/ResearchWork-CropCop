from __future__ import annotations


def accumulation_bucket_sample_count(
    *,
    absolute_batch_index: int,
    batches_per_epoch: int,
    micro_batch_size: int,
    accumulation_steps: int,
    dataset_size: int,
) -> int:
    bucket_start_batch = (int(absolute_batch_index) // int(accumulation_steps)) * int(accumulation_steps)
    bucket_end_batch = min(bucket_start_batch + int(accumulation_steps), int(batches_per_epoch))
    bucket_start_sample = bucket_start_batch * int(micro_batch_size)
    bucket_end_sample = min(bucket_end_batch * int(micro_batch_size), int(dataset_size))
    count = bucket_end_sample - bucket_start_sample
    if count <= 0:
        raise RuntimeError("invalid gradient-accumulation bucket sample count")
    return count
