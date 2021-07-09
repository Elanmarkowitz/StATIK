from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_processing import KGProcessedDataset
from data.left_contiguous_csr import LeftContiguousCSR


class KGGraph:
    def __init__(self, edge_lccsr, relation_lccsr, degrees, indegrees, outdegrees, num_relations, present_entities,
                 num_present_entities):
        self.edge_lccsr: LeftContiguousCSR = edge_lccsr
        self.relation_lccsr: LeftContiguousCSR = relation_lccsr
        self.degrees: np.ndarray = degrees
        self.indegrees: np.ndarray = indegrees
        self.outdegrees: np.ndarray = outdegrees
        self.num_relations: int = num_relations
        self.present_entities: np.ndarray = present_entities
        self.num_entities: int = num_present_entities


class KGBaseDataset:
    def __init__(self, processed_dataset: KGProcessedDataset):
        self.entity_text: List[str] = processed_dataset.entity_text
        self.relation_text: List[str] = processed_dataset.relation_text
        self.entity_feat: np.ndarray = processed_dataset.entity_feat
        self.relation_feat: np.ndarray = processed_dataset.relation_feat
        self.num_entities: int = processed_dataset.num_entities
        self.num_relations: int = processed_dataset.num_relations


class KGLoadableDataset(Dataset):
    def __init__(self, base_dataset: KGBaseDataset, graph: KGGraph, hrt: np.ndarray):
        self.base_dataset = base_dataset
        self.graph = graph
        self.hrt = hrt

    def __len__(self):
        return len(self.hrt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.hrt[idx]


class KGInferenceDataset(KGLoadableDataset):
    def __init__(self, base_dataset: KGBaseDataset, graph: KGGraph, hrt: np.ndarray, h_filter, t_filter):
        super(KGInferenceDataset, self).__init__(base_dataset, graph, hrt)
        self.base_ds = base_dataset
        self.graph = graph
        self.hrt = hrt
        self.h_filter = h_filter
        self.t_filter = t_filter
        self.is_query_mode = True

    def target_mode(self):
        self.is_query_mode = False
        return self

    def query_mode(self):
        self.is_query_mode = True
        return self

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.is_query_mode:
            return self.hrt[idx], self.h_filter[idx], self.t_filter[idx]
        else:
            return idx

    def __len__(self):
        if self.is_query_mode:
            return len(self.hrt)
        else:
            return self.base_dataset.num_entities


def get_training_dataset(ds: KGProcessedDataset) -> KGLoadableDataset:
    base_dataset = KGBaseDataset(ds)
    present_entities = ds.train_entities
    graph = KGGraph(ds.train_edge_lccsr, ds.train_relation_lccsr, ds.train_degrees, ds.train_indegrees,
                    ds.train_outdegrees, ds.num_relations, present_entities, len(present_entities))
    return KGLoadableDataset(base_dataset, graph, ds.train_hrt)


def get_training_inference_dataset(ds: KGProcessedDataset) -> KGInferenceDataset:
    base_dataset = KGBaseDataset(ds)
    present_entities = ds.train_entities
    graph = KGGraph(ds.train_edge_lccsr, ds.train_relation_lccsr, ds.train_degrees, ds.train_indegrees,
                    ds.train_outdegrees, ds.num_relations, present_entities, len(present_entities))
    return KGInferenceDataset(base_dataset, graph, ds.train_hrt, ds.train_h_filter, ds.train_t_filter)


def get_validation_dataset(ds: KGProcessedDataset) -> KGInferenceDataset:
    base_dataset = KGBaseDataset(ds)
    present_entities = np.concatenate([ds.train_entities, ds.valid_entities])
    graph = KGGraph(ds.valid_edge_lccsr, ds.valid_relation_lccsr, ds.valid_degrees, ds.valid_indegrees,
                    ds.valid_outdegrees, ds.num_relations, present_entities, len(present_entities))
    return KGInferenceDataset(base_dataset, graph, ds.valid_hrt, ds.valid_h_filter, ds.valid_t_filter)


def get_testing_dataset(ds: KGProcessedDataset) -> KGInferenceDataset:
    base_dataset = KGBaseDataset(ds)
    present_entities = np.concatenate([ds.train_entities, ds.test_entities])
    graph = KGGraph(ds.test_edge_lccsr, ds.test_relation_lccsr, ds.test_degrees, ds.test_indegrees,
                    ds.test_outdegrees, ds.num_relations, present_entities, len(present_entities))
    return KGInferenceDataset(base_dataset, graph, ds.test_hrt, ds.test_h_filter, ds.test_t_filter)
