# Branch Keshav

import array
from ogb.lsc import WikiKG90MDataset
import torch
from torch.utils.data import Dataset
import random
import numpy as np 

from data import data_processing
from data.left_contiguous_csr import LeftContiguousCSR


class WikiKG90MProcessedDataset(Dataset):
    """WikiKG90M processed dataset."""

    def __init__(self, root_data_dir: str = None, include_inverse: bool = True, from_dataset: WikiKG90MDataset = None):
        """
        Args:
            root_data_dir (str): Root data dir containing the processed WikiKG90M dataset.
        """
        assert root_data_dir is not None or from_dataset is not None, \
            "Must initiate WikiKG90MProcessedDataset with data dir or from_dataset."
        if from_dataset:
            dataset = from_dataset
        else:
            dataset = data_processing.load_processed_data(root_data_dir)
        self.num_entities = dataset.num_entities
        self.train_ht = dataset.train_ht
        self.train_r = dataset.train_r
        self.num_training_relations = dataset.num_relations
        self.num_relations = dataset.num_relations
        if include_inverse:
            self.includes_inverse = True
            self.num_relations_both = dataset.num_relations_both
            self.train_ht_both = dataset.train_ht_both # ht pairs for training
            self.train_r_both = dataset.train_r_both # corresponding relation
        else:
            self.includes_inverse = False
        self.entity_feat = dataset.entity_feat  # feature matrix for entities
        self.relation_feat = dataset.relation_feat  # feature matrix for relation types
        self.edge_lccsr: LeftContiguousCSR = dataset.edge_lccsr
        self.relation_lccsr: LeftContiguousCSR = dataset.relation_lccsr
        self.degrees = dataset.degrees
        self.feature_dim = self.entity_feat.shape[1]
        self.valid_dict = dataset.valid_dict
        self.test_dict = dataset.test_dict

    @staticmethod
    def init_post_processing(processed_dataset):
        """Load directly from results of data_processing.load_processed_data
        Primarily used for development purposes.
        """
        return WikiKG90MProcessedDataset("", from_dataset=processed_dataset)

    def __len__(self):
        return len(self.train_ht)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_ht = self.train_ht[idx]
        batch_r = self.train_r[idx]

        return batch_ht, batch_r

    #@profile
    def add_neighbors_for_batch(self, entity_set: set, edge_heads: array, edge_tails: array, edge_relations: array,
                                is_query: array, node, ignore_r, ignore_t, max_neighbors: int):
        # TODO: inverse relation should load as original relation in reverse order. Let Model handle inverse
        if self.degrees[node] > max_neighbors:
            selection = np.random.randint(self.degrees[node], size=(max_neighbors,))
            tails = self.edge_lccsr[node][selection]
            rels = self.relation_lccsr[node][selection]
        else:
            tails = self.edge_lccsr[node]
            rels = self.relation_lccsr[node]
        non_ignored_idx = np.logical_or(tails != ignore_t, rels != ignore_r) # do not include query relation here
        rels = rels[non_ignored_idx]
        tails = tails[non_ignored_idx]
        inverse_relation_idx = rels >= self.num_relations
        forward_relation_idx = np.logical_not(inverse_relation_idx)

        for t in tails:
            entity_set.add(t)

        count_forward = forward_relation_idx.sum()
        edge_heads.extend(np.repeat(node, count_forward))
        edge_tails.extend(tails[forward_relation_idx])
        edge_relations.extend(rels[forward_relation_idx])

        count_backward = inverse_relation_idx.sum()
        edge_heads.extend(tails[inverse_relation_idx])
        edge_tails.extend(np.repeat(node, count_backward))
        edge_relations.extend(rels[inverse_relation_idx] - self.num_relations)

        is_query.extend(np.repeat(0, count_forward + count_backward))

        # zipped_itr = zip(tails, rels)
        # for tail, rel in zipped_itr:
        #     entity_set.add(tail)
        #     if rel > self.num_relations:  # inverse relation
        #         edge_heads.append(tail)
        #         edge_tails.append(node)
        #         edge_relations.append(rel - self.num_relations)
        #     else:
        #         edge_heads.append(node)
        #         edge_tails.append(tail)
        #         edge_relations.append(rel)
        #     is_query.append(0)
        return

    def get_collate_fn(self, single_query_per_head: bool = True, max_neighbors: int = 10, read_memmap: bool = False,
                       eval_mode: bool = False):
        #@profile
        def wikikg_collate_fn(batch):
            # query edge marked as query
            # 1-hop connected entities included

            entity_set = set()
            edge_heads = array.array("i")
            edge_tails = array.array("i")
            edge_relations = array.array("i")
            is_query = array.array("i")
            labels = array.array("i")

            def add_neighbors(node, ignore_r, ignore_t):
                return self.add_neighbors_for_batch(entity_set, edge_heads, edge_tails, edge_relations, is_query, node,
                                                    ignore_r, ignore_t, max_neighbors)

            for _ht, _r in batch:
                # get neighbors of h and t
                _h, _t = _ht[0], _ht[1]
                _label = 1
                
                if single_query_per_head and not eval_mode:
                    if random.choice([True, False]):
                        _t = random.randrange(self.num_entities)
                        _label = 0

                add_neighbors(_h, _r, _t)
                add_neighbors(_t, _r + self.num_relations, _h)

                # add positive query
                entity_set.add(_h)
                entity_set.add(_t)
                edge_heads.append(_h)
                edge_tails.append(_t)
                edge_relations.append(_r)
                is_query.append(1)
                labels.append(_label)

                # add negative sample
                if not single_query_per_head and not eval_mode:
                    neg_t = random.randrange(self.num_entities)
                    add_neighbors(neg_t, _r + self.num_relations, _h)
                    # add negative query
                    entity_set.add(neg_t)
                    edge_heads.append(_h)
                    edge_tails.append(neg_t)
                    edge_relations.append(_r)
                    is_query.append(1)
                    labels.append(0)

            ht_tensor = torch.from_numpy(np.stack([edge_heads, edge_tails]).transpose()).long()
            r_tensor = torch.from_numpy(np.array(edge_relations)).long()
            entity_set = torch.tensor(list(entity_set)).long()
            entity_feat = torch.from_numpy(self.entity_feat[entity_set]).float() if read_memmap else None
            queries = torch.from_numpy(np.array(is_query)).long()
            labels = torch.from_numpy(np.array(labels)).long()
            node_id_to_batch = torch.empty((self.num_entities,), dtype=torch.int64)
            node_id_to_batch[entity_set] = torch.arange(entity_set.shape[0])
            ht_tensor_batch = node_id_to_batch[ht_tensor]
            return ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, node_id_to_batch, queries, labels
        return wikikg_collate_fn


def load_dataset(root_data_dir: str):
    return WikiKG90MProcessedDataset(root_data_dir)


class Wiki90MEvaluationDataset(Dataset):

    def __init__(self, full_dataset: WikiKG90MProcessedDataset, task):
        self.ds = full_dataset
        self.task = task
        self.hr = task['hr']
        self.t_candidate = task['t_candidate']
        self.t_correct_index = None

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_h = self.hr[idx][0]
        batch_r = self.hr[idx][1]
        batch_t_candidate = self.t_candidate[idx]
        t_correct_idx = self.t_correct_index[idx] if self.t_correct_index is not None else None

        return batch_h, batch_r, batch_t_candidate, t_correct_idx

    def sub_batch_loader(self, batch):
        pass

    def get_eval_collate_fn(self, single_query_per_head=True, max_neighbors=10, read_memmap=False):
        def collate_fn(batch):
            hrt_collate = self.ds.get_collate_fn(single_query_per_head=single_query_per_head, max_neighbors=max_neighbors,
                                                 read_memmap=read_memmap, eval_mode=True)
            subbatches = []
            for i in range(self.t_candidate.shape[1]):
                subbatch_ht = []
                subbatch_r = []
                for _h, _r, _t_candidates, _ in batch:
                    subbatch_ht.append(np.array([_h, _t_candidates[i]]))
                    subbatch_r.append(_r)
                subbatch = hrt_collate(zip(subbatch_ht, subbatch_r))
                subbatches.append(subbatch)

            t_correct_idx = None
            if isinstance(self, Wiki90MValidationDataset):  # t_correct_index exists
                t_correct_idx = []
                for _, _, _, _t_correct_idx in batch:
                    t_correct_idx.append(_t_correct_idx)
                t_correct_idx = np.array(t_correct_idx)

            return subbatches, t_correct_idx

        return collate_fn


class Wiki90MValidationDataset(Wiki90MEvaluationDataset):
    def __init__(self, full_dataset: WikiKG90MProcessedDataset):
        super(Wiki90MValidationDataset, self).__init__(full_dataset, full_dataset.valid_dict['h,r->t'])
        self.t_correct_index = self.task['t_correct_index']


class Wiki90MTestDataset(Wiki90MEvaluationDataset):
    def __init__(self, full_dataset: WikiKG90MProcessedDataset):
        super(Wiki90MTestDataset, self).__init__(full_dataset, full_dataset.test_dict['h,r->t'])



