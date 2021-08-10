from collections import defaultdict

import numpy as np

from data.data_classes import KGGraph, KGInferenceDataset


def compute_eval_stats(pos_scores: np.ndarray, target_scores: np.ndarray, filter_mask: np.ndarray = None):
    better_than_correct_indices = target_scores > pos_scores.reshape(-1, 1)
    results = compute_eval_stats_raw(better_than_correct_indices)
    if filter_mask is not None:
        better_than_correct_indices_filtered = np.logical_and(better_than_correct_indices, filter_mask)
        filtered_results = compute_eval_stats_raw(better_than_correct_indices_filtered)
        for k, v in filtered_results.items():
            results[k + '_filtered'] = v
    return results


def compute_eval_stats_raw(better_than_correct_indices: np.ndarray):
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


def create_filter_mask_for_transfer(inference_dataset: KGInferenceDataset, queries, head_pred=False):
    graph: KGGraph = inference_dataset.graph
    candidate_filter = np.ones((queries.shape[0], graph.num_entities), dtype=np.bool)
    node2idx = -1 * np.ones((inference_dataset.base_ds.num_entities,), dtype=np.int)
    node2idx[graph.present_entities] = np.arange(graph.num_entities)
    for i in range(queries.shape[0]):
        s, r = queries[i]
        if head_pred:
            r = r + graph.num_relations
        to_mask = graph.edge_lccsr[s][graph.relation_lccsr[s] == r]
        candidate_filter[i, to_mask] = False
    return candidate_filter