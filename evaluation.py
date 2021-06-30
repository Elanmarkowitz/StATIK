
import numpy as np


def compute_eval_stats(pos_scores: np.ndarray, target_scores: np.ndarray, filter_mask: np.ndarray = None):
    better_than_correct_indices = target_scores > pos_scores.reshape(-1, 1)
    if filter_mask is not None:
        better_than_correct_indices = np.logical_and(better_than_correct_indices, filter_mask)
    rank = 1 + better_than_correct_indices.sum(1)
    reciprocal_rank = 1 / rank
    mean_rank = rank.mean()
    mean_reciprocal_rank = reciprocal_rank.mean()
    hits_at_1 = (rank == 1).mean()
    hits_at_3 = (rank <= 3).mean()
    hits_at_10 = (rank <= 10).mean()
    return {
        'mr': mean_rank,
        'mrr': mean_reciprocal_rank,
        'hits_at_1': hits_at_1,
        'hits_at_3': hits_at_3,
        'hits_at_10': hits_at_10
    }
