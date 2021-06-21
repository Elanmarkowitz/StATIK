from array import array
from ogb.lsc import WikiKG90MDataset
import torch
from torch.utils.data import Dataset
import numpy as np 

from data import data_processing
from data.left_contiguous_csr import LeftContiguousCSR


class KGProcessedDataset(Dataset):
    """WikiKG90M processed dataset."""

    def __init__(self, root_data_dir: str = None, dataset_name: str = None, from_dataset: WikiKG90MDataset = None):
        """
        Args:
            root_data_dir (str): Root data dir containing the processed WikiKG90M dataset.
        """
        assert (root_data_dir is not None and dataset_name) is not None or from_dataset is not None, \
            "Must initiate KGProcessedDataset with (data dir + dataset name) or from_dataset."
        if from_dataset:
            dataset = from_dataset
        else:
            dataset = data_processing.load_processed_data(root_data_dir, dataset_name)
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
        self.indegrees = dataset.indegrees
        self.outdegrees = dataset.outdegrees
        self.feature_dim = self.entity_feat.shape[1]
        self.valid_dict = dataset.valid_dict
        self.test_dict = dataset.test_dict
        self.access_to_full_graph = False
        self.training_entities = np.unique(self.train_ht)
        if hasattr(dataset, 'valid_edge_lccsr'):
            self.access_to_full_graph = True
            self.valid_edge_lccsr = dataset.valid_edge_lccsr
            self.valid_relation_lccsr = dataset.valid_relation_lccsr
            self.test_edge_lccsr = dataset.test_edge_lccsr
            self.test_relation_lccsr = dataset.test_relation_lccsr
            entities = np.zeros((self.num_entities,), dtype=np.int32)
            validation_entities = np.unique(dataset.valid_hrt[:, [0, 2]])
            entities[validation_entities] = 1
            entities[self.training_entities] = 0
            self.validation_entities = np.nonzero(entities)[0]
            entities = np.zeros((self.num_entities,), dtype=np.int32)
            test_entities = np.unique(dataset.test_hrt[:, [0, 2]])
            entities[test_entities] = 1
            entities[self.training_entities] = 0
            entities[self.validation_entities] = 0
            self.test_entities = np.nonzero(entities)[0]

    @staticmethod
    def init_post_processing(processed_dataset):
        """Load directly from results of data_processing.load_processed_data
        Primarily used for development purposes.
        """
        return KGProcessedDataset("", from_dataset=processed_dataset)

    def __len__(self):
        return len(self.train_ht)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_ht = self.train_ht[idx]
        batch_r = self.train_r[idx]

        return batch_ht[0], batch_r, np.array([batch_ht[1]])

    def sample_neighbors(self, node, max_neighbors: int, mode: str):
        if mode == "train" or not self.access_to_full_graph:
            edge_lccsr = self.edge_lccsr
            relation_lccsr = self.relation_lccsr
        elif mode == "valid":
            edge_lccsr = self.valid_edge_lccsr
            relation_lccsr = self.valid_relation_lccsr
        else:  # mode == "test"
            edge_lccsr = self.test_edge_lccsr
            relation_lccsr = self.test_relation_lccsr
        if self.degrees[node] > max_neighbors:
            selection = np.random.randint(self.degrees[node], size=(max_neighbors,))
            tails = edge_lccsr[node][selection]
            rels = relation_lccsr[node][selection]
        else:
            tails = edge_lccsr[node]
            rels = relation_lccsr[node]
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

    def get_collate_fn(self, max_neighbors: int = 10, sample_negs: int = 0, neg_heads=False, mode="train"):
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

            neg_candidates = np.empty((len(batch), 0), dtype=np.int64)
            neg_h_candiates = np.empty((len(batch), 0), dtype=np.int64)
            if sample_negs:
                if mode == 'train':
                    neg_candidates = self.training_entities[np.random.randint(0, len(self.training_entities), size=(len(batch), sample_negs))]
                else:
                    neg_candidates = np.random.randint(0, self.num_entities, size=(len(batch), sample_negs))
                if neg_heads:
                    if mode == 'train':
                        neg_h_candiates = self.training_entities[np.random.randint(0, len(self.training_entities), size=(len(batch), sample_negs))]
                    else:
                        neg_h_candiates = np.random.randint(0, self.num_entities, size=(len(batch), sample_negs))

            for (_h, _r, _ts), _neg_ts, _neg_hs in zip(batch, neg_candidates, neg_h_candiates):
                t_candidates = np.concatenate([_ts, _neg_ts])
                sample_labels = np.concatenate([np.ones_like(_ts), np.zeros_like(_neg_ts)])
                true_t = _ts[0]

                rels_h, tails_h = self.sample_neighbors(_h, max_neighbors, mode=mode)

                for _t, _label in zip(t_candidates, sample_labels):

                    rels_t, tails_t = self.sample_neighbors(_t, max_neighbors, mode=mode)

                    rels_h_t, tails_h_t = self.ignore_query(rels_h, tails_h, _r, _t)
                    rels_t, tails_t = self.ignore_query(rels_t, tails_t, _r + self.num_relations, _h)

                    component, c_size = self.create_component(_h, rels_h_t, tails_h_t, _t, rels_t, tails_t, _r, _label)

                    self.add_component(edge_heads, edge_relations, edge_tails, is_query, labels, cumulative_entities,
                                       batch_id_to_node_id, r_queries, r_relatives, h_or_t_sample, component)
                    cumulative_entities += c_size

                if neg_heads:
                    rels_t, tails_t = self.sample_neighbors(true_t, max_neighbors, mode=mode)
                    for _h, _label in zip(_neg_hs, np.zeros_like(_neg_hs)):
                        rels_h, tails_h = self.sample_neighbors(_h, max_neighbors, mode=mode)

                        rels_t_h, tails_t_h = self.ignore_query(rels_t, tails_t, _r + self.num_relations, _h)
                        rels_h, tails_h = self.ignore_query(rels_h, tails_h, _r, true_t)

                        component, c_size = self.create_component(_h, rels_h, tails_h, true_t, rels_t_h, tails_t_h, _r,
                                                                  _label)

                        self.add_component(edge_heads, edge_relations, edge_tails, is_query, labels,
                                           cumulative_entities,
                                           batch_id_to_node_id, r_queries, r_relatives, h_or_t_sample, component)
                        cumulative_entities += c_size

            ht_tensor = torch.from_numpy(np.stack([edge_heads, edge_tails]).transpose()).long()
            r_tensor = torch.from_numpy(np.array(edge_relations)).long()
            entity_set = torch.from_numpy(np.array(batch_id_to_node_id)).long()

            indeg = self.indegrees[entity_set]
            outdeg = self.outdegrees[entity_set]
            indeg_feat = np.stack([indeg // 10**i for i in range(0, 7)]).astype(np.bool).astype(np.int32).T
            outdeg_feat = np.stack([outdeg // 10 ** i for i in range(0, 7)]).astype(np.bool).astype(np.int32).T
            indeg_feat = torch.from_numpy(indeg_feat).float()
            outdeg_feat = torch.from_numpy(outdeg_feat).float()

            entity_feat = None  # TODO: Remove this
            queries = torch.from_numpy(np.array(is_query)).long()
            labels = torch.from_numpy(np.array(labels)).long()
            r_queries = torch.from_numpy(np.array(r_queries)).long()
            r_relatives = torch.from_numpy(np.array(r_relatives)).long()
            h_or_t_sample = torch.from_numpy(np.array(h_or_t_sample)).long()
            return ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
        return wikikg_collate_fn


def load_dataset(root_data_dir: str, dataset_name: str):
    return KGProcessedDataset(root_data_dir=root_data_dir, dataset_name=dataset_name)


class KGEvaluationDataset(Dataset):

    def __init__(self, full_dataset: KGProcessedDataset, task, num_candidates_per_itr=None, head_prediction=False):
        self.ds = full_dataset
        self.task = task
        if head_prediction:  # use tails as "head" and head candidates as "t candidates". Reverse after collate
            self.hr = task['tr']
            self.t_candidate = task['h_candidate']
            self.t_candidate_filter_mask = None
            if 'h_candidate_filter_mask' in task:
                self.t_candidate_filter_mask = task['h_candidate_filter_mask']
            self.t_correct_index = None
        else:
            self.hr = task['hr']
            self.t_candidate = task['t_candidate']
            self.t_candidate_filter_mask = None
            if 't_candidate_filter_mask' in task:
                self.t_candidate_filter_mask = task['t_candidate_filter_mask']
            self.t_correct_index = None
        self.num_candidates_per_itr = num_candidates_per_itr
        self.head_prediction = head_prediction

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_h = self.hr[idx][0]
        batch_r = self.hr[idx][1]
        batch_t_candidate = self.t_candidate[idx]
        batch_t_candidate_filter_mask = self.t_candidate_filter_mask[idx] if self.t_candidate_filter_mask is not None else None
        t_correct_idx = self.t_correct_index[idx] if self.t_correct_index is not None else None

        return batch_h, batch_r, batch_t_candidate, batch_t_candidate_filter_mask, t_correct_idx

    def sub_batch_loader(self, batch):
        pass

    def get_eval_collate_fn(self, max_neighbors=10):
        def collate_fn(batch):
            mode = "valid" if isinstance(self, KGValidationDataset) else "test"
            hrt_collate = self.ds.get_collate_fn(max_neighbors=max_neighbors, mode=mode)
            batch_h = array("i")
            batch_r = array("i")
            batch_t_candidates = []
            batch_t_filter_masks = []
            t_correct_idx = array("i")
            for _h, _r, _t_candidates, _t_candidates_filter_mask, _t_correct in batch:
                batch_h.append(_h)
                batch_r.append(_r)
                batch_t_candidates.append(_t_candidates)
                if _t_candidates_filter_mask is not None:
                    batch_t_filter_masks.append(_t_candidates_filter_mask)
                if _t_correct is not None:
                    t_correct_idx.append(_t_correct)

            out_batches = []
            if self.num_candidates_per_itr is not None:
                for i in range(0, batch_t_candidates[0].shape[0], self.num_candidates_per_itr):
                    subbatch_t_candidates = [batch_t_candidates[e][i:i+self.num_candidates_per_itr] for e in range(len(batch_t_candidates))]
                    subbatch = hrt_collate(list(zip(batch_h, batch_r, subbatch_t_candidates)))
                    if self.head_prediction:  # reverse for when doing head prediction
                        ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = subbatch
                        ht_tensor = ht_tensor[:, [1, 0]]
                        h_or_t_sample = torch.where(h_or_t_sample == 0, 1, 0)
                        subbatch = ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
                    out_batches.append(subbatch)
            else:
                out_batches.append(hrt_collate(list(zip(batch_h, batch_r, batch_t_candidates))))

            t_correct_idx = np.array(t_correct_idx) if self.t_correct_index is not None else None
            batch_t_filter_masks = np.array(batch_t_filter_masks) if self.t_candidate_filter_mask is not None else None

            return out_batches, t_correct_idx, batch_t_filter_masks

        return collate_fn


class KGValidationDataset(KGEvaluationDataset):
    def __init__(self, full_dataset: KGProcessedDataset, num_candidates_per_itr=1001, head_prediction=False):
        super(KGValidationDataset, self).__init__(full_dataset, full_dataset.valid_dict['h,r->t'],
                                                  num_candidates_per_itr=num_candidates_per_itr,
                                                  head_prediction=head_prediction)
        if head_prediction:
            self.t_correct_index = self.task['h_correct_index']
        else:
            self.t_correct_index = self.task['t_correct_index']


class KGTestDataset(KGEvaluationDataset):
    def __init__(self, full_dataset: KGProcessedDataset, num_candidates_per_itr=1001, head_prediction=False):
        super(KGTestDataset, self).__init__(full_dataset, full_dataset.test_dict['h,r->t'],
                                            num_candidates_per_itr=num_candidates_per_itr,
                                            head_prediction=head_prediction)
        if 't_correct_index' in self.task:
            if head_prediction:
                self.t_correct_index = self.task['h_correct_index']
            else:
                self.t_correct_index = self.task['t_correct_index']



