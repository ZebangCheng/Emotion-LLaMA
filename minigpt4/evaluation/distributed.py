"""Distributed collection for evaluation records."""

import torch.distributed as dist

from minigpt4.evaluation.evaluator import merge_rank_records


def gather_evaluation_records(records):
    """Gather rank-local records on every rank, then remove sampler padding."""
    local_records = list(records)
    if not dist.is_available() or not dist.is_initialized():
        return merge_rank_records([local_records])

    rank_records = [None] * dist.get_world_size()
    dist.all_gather_object(rank_records, local_records)
    return merge_rank_records(rank_records)
