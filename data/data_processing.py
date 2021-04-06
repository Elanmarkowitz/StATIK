from collections import defaultdict
import os
import numpy as np 
import pickle
import scipy.sparse as sp
import array

import tqdm
from ogb.lsc import WikiKG90MDataset

from data.left_contiguous_csr import LeftContiguousCSR

DATA_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/nas/home/elanmark/data"


def load_original_data(root_data_dir: str) -> WikiKG90MDataset:
    dataset = WikiKG90MDataset(root=root_data_dir)
    return dataset


def process_data(root_data_dir: str) -> None:
    print('Loading original data.')
    dataset = load_original_data(root_data_dir)
    save_dir = os.path.join(root_data_dir, "wikikg90m_kddcup2021", "processed")

    # separate indices and relations
    train_hrt = dataset.train_hrt
    train_ht = train_hrt[:, [0,2]]
    train_r = train_hrt[:,1]

    # add inverse relations
    print("Adding inverse relations.")
    train_ht_inverse = train_ht[:,[1,0]]
    train_r_inverse = train_r + dataset.num_relations
    train_ht_both = np.concatenate([train_ht, train_ht_inverse], axis=0)
    train_r_both = np.concatenate([train_r, train_r_inverse], axis=0)

    np.save(os.path.join(save_dir, 'train_ht_inverse.npy'), train_ht_inverse)
    np.save(os.path.join(save_dir, 'train_r_inverse.npy'), train_r_inverse)
    np.save(os.path.join(save_dir, 'train_ht_both.npy'), train_ht_both)
    np.save(os.path.join(save_dir, 'train_r_both.npy'), train_r_both)

    # dictionary of edges
    edge_dict = defaultdict(lambda: array.array('i')) # sp.dok_matrix((dataset.num_entities, dataset.num_entities), dtype=np.int64)
    relation_dict = defaultdict(lambda: array.array('i')) # sp.dok_matrix((dataset.num_entities, dataset.num_entities), dtype=np.int64)
    degrees = np.zeros((dataset.num_entities,), dtype=np.int64)
    degrees = np.load(os.path.join(save_dir, 'degrees.npy'))
    with open(os.path.join(save_dir, 'edge_dict.pkl'), 'rb') as f:
        edge_dict = pickle.load(f)
    with open(os.path.join(save_dir, 'relation_dict.pkl'), 'rb') as f:
        relation_dict = pickle.load(f)
    print("Building edge dict.")
    for i in tqdm.tqdm(range(len(train_ht))):
        h,r,t,r_inv = int(train_ht[i][0]), int(train_r[i]), int(train_ht[i][1]), int(train_r_inverse[i])
        edge_dict[h].append(t)
        relation_dict[h].append(r)
        degrees[h] = degrees[h] + 1
        edge_dict[t].append(h)
        relation_dict[t].append(r_inv)
        degrees[t] = degrees[t] + 1
    edge_dict = dict(edge_dict)
    relation_dict = dict(relation_dict)
    print("Converting to np arrays.")
    edge_csr_data = np.zeros((2 * len(train_ht),), dtype=np.int32)
    edge_csr_indices = np.zeros((2 * len(train_ht),), dtype=np.int32)
    edge_csr_indptr = np.zeros((dataset.num_entities + 1,))
    rel_csr_data = np.zeros((2 * len(train_ht),), dtype=np.int32)
    rel_csr_indices = np.zeros((2 * len(train_ht),), dtype=np.int32)
    rel_csr_indptr = np.zeros((dataset.num_entities + 1,))
    num_prev = 0
    for i in tqdm.tqdm(range(dataset.num_entities)):
        deg = degrees[i]
        edge_csr_indptr[i] = num_prev
        edge_csr_indices[num_prev:num_prev+deg] = np.arange(0, deg)
        edge_csr_data[num_prev:num_prev+deg] = np.array(edge_dict[i], dtype=np.int32)
        rel_csr_indptr[i] = num_prev
        rel_csr_indices[num_prev:num_prev + deg] = np.arange(0, deg)
        rel_csr_data[num_prev:num_prev + deg] = np.array(relation_dict[i], dtype=np.int32)
        num_prev += degrees[i]
    edge_csr_indptr[-1] = num_prev
    rel_csr_indptr[-1] = num_prev
    # edge_csr = sp.csr_matrix((edge_csr_data, edge_csr_indices, edge_csr_indptr),
    #                          shape=(dataset.num_entities, dataset.num_entities))
    # rel_csr = sp.csr_matrix((rel_csr_data, rel_csr_indices, rel_csr_indptr),
    #                         shape=(dataset.num_entities, dataset.num_entities))
    # sp.save_npz(os.path.join(save_dir, 'edge_csr.npz'), edge_csr)
    # sp.save_npz(os.path.join(save_dir, 'rel_csr.npz'), rel_csr)

    rel_lccsr = LeftContiguousCSR(rel_csr_indptr, degrees, rel_csr_data)
    edge_lccsr = LeftContiguousCSR(edge_csr_indptr, degrees, edge_csr_data)
    rel_lccsr.save(os.path.join(save_dir, 'rel_lccsr.npz'))
    edge_lccsr.save(os.path.join(save_dir, 'edge_lccsr.npz'))

    # for i in tqdm.tqdm(range(dataset.num_entities)):
    #     edge_dict[i] = np.array(edge_dict[i], dtype=np.int32)
    #     relation_dict[i] = np.array(relation_dict[i], dtype=np.int32)


    # print('Saving edge dict.')
    # with open(os.path.join(save_dir, 'edge_dict.pkl'), 'wb') as f:
    #     pickle.dump(edge_dict, f)
    # print('Saving relation dict.')
    # with open(os.path.join(save_dir, 'relation_dict.pkl'), 'wb') as f:
    #     pickle.dump(relation_dict, f)
    np.save(os.path.join(save_dir, 'degrees.npy'), degrees)


def load_processed_data(root_data_dir: str) -> WikiKG90MDataset:
    save_dir = os.path.join(root_data_dir, "wikikg90m_kddcup2021", "processed")
    print('Loading processed dataset.')
    dataset = load_original_data(root_data_dir)
    dataset.degrees = np.load(os.path.join(save_dir, 'degrees.npy'))
    dataset.train_ht_inverse = np.load(os.path.join(save_dir, 'train_ht_inverse.npy'))
    dataset.train_r_inverse = np.load(os.path.join(save_dir, 'train_r_inverse.npy'))
    dataset.train_ht_both = np.load(os.path.join(save_dir, 'train_ht_both.npy'))
    dataset.train_r_both = np.load(os.path.join(save_dir, 'train_r_both.npy'))
    dataset.train_ht = dataset.train_hrt[:, [0,2]]
    dataset.train_r = dataset.train_hrt[:,1]
    dataset.num_relations_both = dataset.num_relations * 2
    # print('Loading edge dict.')
    # with open(os.path.join(save_dir, 'edge_dict.pkl'), 'rb') as f:
    #     dataset.edge_dict = pickle.load(f)
    # print('Loading relation dict.')
    # with open(os.path.join(save_dir, 'relation_dict.pkl'), 'rb') as f:
    #     dataset.relation_dict = pickle.load(f)
    # dataset.edge_csr = sp.load_npz(os.path.join(save_dir, 'edge_csr.npz'))
    # dataset.relation_csr = sp.load_npz(os.path.join(save_dir, 'rel_csr.npz'))
    dataset.edge_lccsr = LeftContiguousCSR.load(os.path.join(save_dir, 'edge_lccsr.npz'))
    dataset.relation_lccsr = LeftContiguousCSR.load(os.path.join(save_dir, 'rel_lccsr.npz'))
    return dataset


if __name__ == "__main__":
    process_data(DATA_DIR)
