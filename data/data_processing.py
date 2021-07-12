from collections import defaultdict
import os
import numpy as np
import array
from typing import Union

import tqdm
from ogb.lsc import WikiKG90MDataset

from data.left_contiguous_csr import LeftContiguousCSR
from data.wordnet_processing_v2 import ProcessWordNet
from data.fb15k_processing_v2 import ProcessFreebase

DATA_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/elanmark"


LEGAL_DATASETS = {
    "wikikg90m_kddcup2021",
    "FB15k-237",
    "WN18RR"
}

ProcessableDataset = Union[WikiKG90MDataset, ProcessWordNet, ProcessFreebase]


def load_original_data(root_data_dir: str, dataset_name: str) -> ProcessableDataset:
    assert dataset_name in LEGAL_DATASETS, f'DATASET must be one of {list(LEGAL_DATASETS)}'

    if dataset_name == "wikikg90m_kddcup2021":
        return WikiKG90MDataset(root=root_data_dir)
    elif dataset_name == "WN18RR":
        return ProcessWordNet(root_data_dir=root_data_dir)
    elif dataset_name == "FB15k-237":
        return ProcessFreebase(root_data_dir=root_data_dir)
    else:
        raise Exception('Dataset not known.')


def get_filtered_candidate(queries, true_triples, num_entities):
    candidates = np.tile(np.arange(0, num_entities), queries.shape[0])
    A = np.append(np.repeat(queries, num_entities, axis=0), candidates[:, np.newaxis], axis=1)
    dt = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    idx = np.nonzero(np.in1d(A.view(dt).reshape(-1), np.ascontiguousarray(true_triples).view(dt).reshape(-1)))[0]
    candidates[idx] = -1
    return candidates


def get_filtered_candidate_with_lccsr(queries, edge_lccsr, relation_lccsr, num_entities, num_relations, head_pred=True):
    candidate_filter = np.ones((queries.shape[0], num_entities), dtype=np.bool)
    for i in range(queries.shape[0]):
        s, r = queries[i]
        if head_pred:
            r = r + num_relations
        to_mask = edge_lccsr[s][relation_lccsr[s] == r]
        candidate_filter[i, to_mask] = False
    return candidate_filter


def process_data(root_data_dir: str, dataset_name: str) -> None:
    print('Loading original data.')
    dataset = load_original_data(root_data_dir, dataset_name)
    save_dir = os.path.join(root_data_dir, dataset_name, "processed")

    train_hrt = dataset.train_hrt

    def create_lccsr(num_entities, num_edges, hs, rs, ts, r_invs):
        # dictionary of edges
        edge_dict = defaultdict(lambda: array.array('i'))
        relation_dict = defaultdict(lambda: array.array('i'))
        degrees = np.zeros((num_entities,), dtype=np.int32)
        indegrees = np.zeros((num_entities,), dtype=np.int32)
        outdegrees = np.zeros((num_entities,), dtype=np.int32)
        print("Building edge dict.")
        for i in tqdm.tqdm(range(num_edges)):
            h,r,t,r_inv = int(hs[i]), int(rs[i]), int(ts[i]), int(r_invs[i])
            edge_dict[h].append(t)
            relation_dict[h].append(r)
            degrees[h] = degrees[h] + 1
            outdegrees[h] = outdegrees[h] + 1

            edge_dict[t].append(h)
            relation_dict[t].append(r_inv)
            degrees[t] = degrees[t] + 1
            indegrees[t] = indegrees[t] + 1
        print("Converting to np arrays.")
        edge_csr_data = np.zeros((2 * num_edges,), dtype=np.int32)
        edge_csr_indptr = np.zeros((num_entities + 1,), dtype=np.int32)
        rel_csr_data = np.zeros((2 * num_edges,), dtype=np.int16)
        rel_csr_indptr = np.zeros((num_entities + 1,), dtype=np.int32)
        num_prev = 0
        for i in tqdm.tqdm(range(num_entities)):
            deg = degrees[i]
            edge_csr_indptr[i] = num_prev
            edge_csr_data[num_prev:num_prev+deg] = np.array(edge_dict[i], dtype=np.int32)
            rel_csr_indptr[i] = num_prev
            rel_csr_data[num_prev:num_prev + deg] = np.array(relation_dict[i], dtype=np.int16)
            num_prev += degrees[i]
        edge_csr_indptr[-1] = num_prev
        rel_csr_indptr[-1] = num_prev

        rel_lccsr = LeftContiguousCSR(rel_csr_indptr, degrees, rel_csr_data)
        edge_lccsr = LeftContiguousCSR(edge_csr_indptr, degrees, edge_csr_data)
        return rel_lccsr, edge_lccsr, degrees, indegrees, outdegrees

    total_hrt = np.concatenate((dataset.train_hrt, dataset.valid_hrt, dataset.test_hrt), axis=0)
    for stage, hrt_group in zip(['train', 'valid', 'test'],
                                [train_hrt, np.concatenate((dataset.train_hrt, dataset.valid_hrt), axis=0), total_hrt]):

        rel_lccsr, edge_lccsr, degrees, indegrees, outdegrees = create_lccsr(dataset.num_entities, len(hrt_group),
                                                                             hrt_group[:,0], hrt_group[:,1], hrt_group[:,2],
                                                                             hrt_group[:,1] + dataset.num_relations)
        rel_lccsr.save(os.path.join(save_dir, f'{stage}_rel_lccsr.npz'))
        edge_lccsr.save(os.path.join(save_dir, f'{stage}_edge_lccsr.npz'))
        np.save(os.path.join(save_dir, f'{stage}_degrees.npy'), degrees)
        np.save(os.path.join(save_dir, f'{stage}_indegrees.npy'), indegrees)
        np.save(os.path.join(save_dir, f'{stage}_outdegrees.npy'), outdegrees)

    full_rel_lccsr = LeftContiguousCSR.load(os.path.join(save_dir, f'test_rel_lccsr.npz'))
    full_edge_lccsr = LeftContiguousCSR.load(os.path.join(save_dir, f'test_edge_lccsr.npz'))
    for stage, triples in zip(['train', 'valid', 'test'], [dataset.train_hrt, dataset.valid_hrt, dataset.test_hrt]):
        h_filter_mask = get_filtered_candidate_with_lccsr(triples[:, [2, 1]], full_edge_lccsr, full_rel_lccsr,
                                                          dataset.num_entities, dataset.num_relations, head_pred=True)
        t_filter_mask = get_filtered_candidate_with_lccsr(triples[:, [0, 1]], full_edge_lccsr, full_rel_lccsr,
                                                          dataset.num_entities, dataset.num_relations, head_pred=False)
        np.save(os.path.join(save_dir, f'{stage}_h_filter.npy'), h_filter_mask)
        np.save(os.path.join(save_dir, f'{stage}_t_filter.npy'), t_filter_mask)


class KGProcessedDataset:
    """KG processed dataset."""
    def __init__(self, root_data_dir: str, dataset_name: str):
        load_dir = os.path.join(root_data_dir, dataset_name, "processed")
        print('Loading processed loaded.')
        loaded = load_original_data(root_data_dir, dataset_name)
        self.num_entities = loaded.num_entities
        self.num_relations = loaded.num_relations
        self.entity_feat = loaded.entity_feat
        self.relation_feat = loaded.relation_feat
        self.entity_text = loaded.entity_text
        self.relation_text = loaded.relation_text
        self.train_hrt = loaded.train_hrt
        self.valid_hrt = loaded.valid_hrt
        self.test_hrt = loaded.test_hrt
        self.train_edge_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'train_edge_lccsr.npz'))
        self.train_relation_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'train_rel_lccsr.npz'))
        self.valid_edge_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'valid_edge_lccsr.npz'))
        self.valid_relation_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'valid_rel_lccsr.npz'))
        self.test_edge_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'test_edge_lccsr.npz'))
        self.test_relation_lccsr: LeftContiguousCSR = LeftContiguousCSR.load(os.path.join(load_dir, 'test_rel_lccsr.npz'))
        self.train_degrees = np.load(os.path.join(load_dir, 'train_degrees.npy'))
        self.train_indegrees = np.load(os.path.join(load_dir, 'train_indegrees.npy'))
        self.train_outdegrees = np.load(os.path.join(load_dir, 'train_outdegrees.npy'))
        self.valid_degrees = np.load(os.path.join(load_dir, 'valid_degrees.npy'))
        self.valid_indegrees = np.load(os.path.join(load_dir, 'valid_indegrees.npy'))
        self.valid_outdegrees = np.load(os.path.join(load_dir, 'valid_outdegrees.npy'))
        self.test_degrees = np.load(os.path.join(load_dir, 'test_degrees.npy'))
        self.test_indegrees = np.load(os.path.join(load_dir, 'test_indegrees.npy'))
        self.test_outdegrees = np.load(os.path.join(load_dir, 'valid_outdegrees.npy'))
        self.feature_dim = self.entity_feat.shape[1]
        self.train_h_filter = np.load(os.path.join(load_dir, 'train_h_filter.npy'))
        self.train_t_filter = np.load(os.path.join(load_dir, 'train_t_filter.npy'))
        self.valid_h_filter = np.load(os.path.join(load_dir, 'valid_h_filter.npy'))
        self.valid_t_filter = np.load(os.path.join(load_dir, 'valid_t_filter.npy'))
        self.test_h_filter = np.load(os.path.join(load_dir, 'test_h_filter.npy'))
        self.test_t_filter = np.load(os.path.join(load_dir, 'test_t_filter.npy'))

        self.train_entities = np.unique(self.train_hrt[:, [0, 2]])
        entities = np.zeros((self.num_entities,), dtype=np.int32)
        validation_entities = np.unique(self.valid_hrt[:, [0, 2]])
        entities[validation_entities] = 1
        entities[self.train_entities] = 0
        self.valid_entities = np.nonzero(entities)[0]
        entities = np.zeros((self.num_entities,), dtype=np.int32)
        test_entities = np.unique(self.test_hrt[:, [0, 2]])
        entities[test_entities] = 1
        entities[self.train_entities] = 0
        entities[self.valid_entities] = 0
        self.test_entities = np.nonzero(entities)[0]

        self.train_h_filter[:, np.concatenate([self.valid_entities, self.test_entities])] = False
        self.train_t_filter[:, np.concatenate([self.valid_entities, self.test_entities])] = False
        self.valid_h_filter[:, self.test_entities] = False
        self.valid_t_filter[:, self.test_entities] = False


def load_processed_data(root_data_dir: str, dataset_name: str) -> KGProcessedDataset:
    return KGProcessedDataset(root_data_dir, dataset_name)

