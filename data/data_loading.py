from array import array
from ogb.lsc import WikiKG90MDataset
import torch
from torch.utils.data import Dataset
import numpy as np 

from data import data_processing
from data.left_contiguous_csr import LeftContiguousCSR


class WikiKG90MProcessedDataset(Dataset):
    """WikiKG90M processed dataset."""

    def __init__(self, root_data_dir: str = None, from_dataset: WikiKG90MDataset = None):
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
        self.entity_feat = dataset.entity_feat  # feature matrix for entities
        self.relation_feat = dataset.relation_feat  # feature matrix for relation types   # unused in loader
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

        return batch_ht[0], batch_r, np.array([batch_ht[1]])

    def sample_neighbors(self, node, max_neighbors: int):
        if self.degrees[node] > max_neighbors:
            selection = np.random.randint(self.degrees[node], size=(max_neighbors,))
            tails = self.edge_lccsr[node][selection]
            rels = self.relation_lccsr[node][selection]
        else:
            tails = self.edge_lccsr[node]
            rels = self.relation_lccsr[node]
        return rels, tails

    @staticmethod
    def ignore_query(rels, tails, ignore_r, ignore_t):
        non_ignored_idx = np.logical_or(tails != ignore_t, rels != ignore_r)
        rels = rels[non_ignored_idx]
        tails = tails[non_ignored_idx]
        return rels, tails

    def add_relations_with_inverting(self, node, rels, tails, entity_set: set, edge_heads: array, edge_tails: array,
                                     edge_relations: array):
        inverse_relation_idx = rels >= self.num_relations
        forward_relation_idx = np.logical_not(inverse_relation_idx)
        entity_set.add(node)
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

        return forward_relation_idx

    def create_component(self, h, rels_h, tails_h, t, rels_t, tails_t, r, label):
        entity_set = set()
        edge_heads = array("i")
        edge_tails = array("i")
        edge_relations = array("i")

        fwd_rel_idx_h = self.add_relations_with_inverting(h, rels_h, tails_h, entity_set, edge_heads, edge_tails, edge_relations)
        fwd_rel_idx_t = self.add_relations_with_inverting(t, rels_t, tails_t, entity_set, edge_heads, edge_tails, edge_relations)

        r_relative = np.concatenate([fwd_rel_idx_h, fwd_rel_idx_t, np.array([1])])
        r_query = np.repeat(r, r_relative.shape[0])
        from_head_or_tail = np.concatenate([np.ones(rels_h.shape[0], dtype=np.int32), np.zeros(rels_t.shape[0], dtype=np.int32), np.array([1])])

        entity_set.add(h)
        entity_set.add(t)
        edge_heads.append(h)
        edge_relations.append(r)
        edge_tails.append(t)

        is_query = np.zeros((len(edge_heads),), dtype=np.int64)
        is_query[-1] = 1

        entity_set_list = list(entity_set)
        batch_id_to_node_id = np.array(entity_set_list)
        node_id_to_batch_node_id = dict((e, i) for (i, e) in enumerate(entity_set_list))
        edge_heads = np.array([node_id_to_batch_node_id[e] for e in edge_heads])
        edge_tails = np.array([node_id_to_batch_node_id[e] for e in edge_tails])

        return (edge_heads, edge_relations, edge_tails, is_query, label, batch_id_to_node_id, r_query, r_relative, from_head_or_tail), len(entity_set)

    @staticmethod
    def add_component(edge_heads, edge_relations, edge_tails, is_query, labels, cumulative_entities,
                      batch_id_to_node_id, r_queries, r_relatives, head_or_tail_sample, component):
        c_edge_heads, c_edge_relations, c_edge_tails, c_is_query, c_label, c_batch_id_to_node_id, r_query, r_relative, from_head_or_tail = component

        edge_heads.extend(c_edge_heads + cumulative_entities)
        edge_tails.extend(c_edge_tails + cumulative_entities)
        edge_relations.extend(c_edge_relations)
        is_query.extend(c_is_query)
        labels.append(c_label)
        batch_id_to_node_id.extend(c_batch_id_to_node_id)
        r_queries.extend(r_query)
        r_relatives.extend(r_relative)
        head_or_tail_sample.extend(from_head_or_tail)

        return

    def get_collate_fn(self, max_neighbors: int = 10, sample_negs: int = 0):
        def wikikg_collate_fn(batch):
            # query edge marked as query
            # 1-hop connected entities included
            batch_id_to_node_id = array("i")
            node_id_to_batch_id = np.empty((self.num_entities,), dtype=np.int64)
            edge_heads = array("i")
            edge_tails = array("i")
            edge_relations = array("i")
            is_query = array("i")
            labels = array("i")
            r_queries = array("i")
            r_relatives = array("i")
            h_or_t_sample = array("i")
            cumulative_entities = 0

            if sample_negs:
                neg_candidates = np.random.randint(0, self.num_entities, size=(len(batch), sample_negs))
            else:
                neg_candidates = np.empty((len(batch), 0), dtype=np.int64)

            for (_h, _r, _ts), _neg_ts in zip(batch, neg_candidates):
                t_candidates = np.concatenate([_ts, _neg_ts])
                sample_labels = np.concatenate([np.ones_like(_ts), np.zeros_like(_neg_ts)])

                rels_h, tails_h = self.sample_neighbors(_h, max_neighbors)

                for _t, _label in zip(t_candidates, sample_labels):

                    rels_t, tails_t = self.sample_neighbors(_t, max_neighbors)

                    rels_h_t, tails_h_t = self.ignore_query(rels_h, tails_h, _r, _t)
                    rels_t, tails_t = self.ignore_query(rels_t, tails_t, _r + self.num_relations, _h)

                    component, c_size = self.create_component(_h, rels_h_t, tails_h_t, _t, rels_t, tails_t, _r, _label)

                    self.add_component(edge_heads, edge_relations, edge_tails, is_query, labels, cumulative_entities,
                                       batch_id_to_node_id, r_queries, r_relatives, h_or_t_sample, component)
                    cumulative_entities += c_size

            ht_tensor = torch.from_numpy(np.stack([edge_heads, edge_tails]).transpose()).long()
            r_tensor = torch.from_numpy(np.array(edge_relations)).long()
            entity_set = torch.from_numpy(np.array(batch_id_to_node_id)).long()
            entity_feat = None  # TODO: Remove this
            queries = torch.from_numpy(np.array(is_query)).long()
            labels = torch.from_numpy(np.array(labels)).long()
            r_queries = torch.from_numpy(np.array(r_queries)).long()
            r_relatives = torch.from_numpy(np.array(r_relatives)).long()
            h_or_t_sample = torch.from_numpy(np.array(h_or_t_sample)).long()
            return ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
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

    def get_eval_collate_fn(self, max_neighbors=10):
        def collate_fn(batch):
            hrt_collate = self.ds.get_collate_fn(max_neighbors=max_neighbors)

            batch_h = array("i")
            batch_r = array("i")
            batch_t_candidates = []
            t_correct_idx = array("i")
            for _h, _r, _t_candidates, _t_correct in batch:
                batch_h.append(_h)
                batch_r.append(_r)
                batch_t_candidates.append(_t_candidates)
                t_correct_idx.append(_t_correct)

            out_batch = hrt_collate(list(zip(batch_h, batch_r, batch_t_candidates)))

            t_correct_idx = t_correct_idx if isinstance(self, Wiki90MValidationDataset) else None

            return out_batch, np.array(t_correct_idx)

        return collate_fn


class Wiki90MValidationDataset(Wiki90MEvaluationDataset):
    def __init__(self, full_dataset: WikiKG90MProcessedDataset):
        super(Wiki90MValidationDataset, self).__init__(full_dataset, full_dataset.valid_dict['h,r->t'])
        self.t_correct_index = self.task['t_correct_index']


class Wiki90MTestDataset(Wiki90MEvaluationDataset):
    def __init__(self, full_dataset: WikiKG90MProcessedDataset):
        super(Wiki90MTestDataset, self).__init__(full_dataset, full_dataset.test_dict['h,r->t'])



