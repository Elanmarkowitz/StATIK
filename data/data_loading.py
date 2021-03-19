from ogb.lsc import WikiKG90MDataset
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np 

from data import data_processing


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
        self.edge_dict = dataset.edge_dict  # dict from head id to list of tail ids #includes both pairs at the moment
        self.relation_dict = dataset.relation_dict  # corresponding to edge_dict, dict from head id to list of relation ids #includes both directions at the moment
        self.degrees = dataset.degrees

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

    def get_collate_fn(self, single_query_per_head=True, max_neighbors=10):
        def wikikg_collate_fn(batch):
            # query edge marked as query
            # 1-hop connected entities included

            entity_set = set()
            edge_heads = []
            edge_tails = []
            edge_relations = []
            is_query = []
            labels = []

            def add_neighbors(node, ignore_r, ignore_t):
                if self.degrees[node] > max_neighbors:
                    selection = np.random.choice(np.arange(self.degrees[node]), size=(max_neighbors,), replace=False)
                else:
                    selection = np.arange(self.degrees[node])
                for tail, rel in zip(self.edge_dict[node][selection], self.relation_dict[node][selection]):
                    if tail == ignore_t and rel == ignore_r:  # do not include query relation here
                        continue
                    entity_set.add(tail)
                    edge_heads.append(_h)
                    edge_tails.append(tail)
                    edge_relations.append(rel)
                    is_query.append(0)

            for _ht, _r in batch:
                # get neighbors of h and t
                _h, _t = _ht[0], _ht[1]
                _label = 1
                
                if single_query_per_head:
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
                labels.append(1)

                # add negative sample
                if not single_query_per_head:
                    neg_t = random.randrange(self.num_entities)
                    add_neighbors(neg_t, _r + self.num_relations, _h)
                    # add negative query
                    entity_set.add(neg_t)
                    edge_heads.append(_h)
                    edge_tails.append(neg_t)
                    edge_relations.append(_r)
                    is_query.append(1)
                    labels.append(0)

            ht_tensor = torch.tensor([edge_heads, edge_tails])
            r_tensor = torch.tensor(edge_relations)
            entity_set = torch.tensor(list(entity_set))
            entity_feat = torch.tensor(self.entity_feat[entity_set])
            queries = torch.tensor(is_query)
            labels = torch.tensor(labels)
            node_id_to_batch = -torch.ones((self.num_entities,), dtype=torch.int64)
            node_id_to_batch[entity_set] = torch.arange(entity_set.shape[0])
            return ht_tensor, r_tensor, entity_set, entity_feat, node_id_to_batch, queries, labels
        return wikikg_collate_fn


def load_dataset(root_data_dir: str):
    return WikiKG90MProcessedDataset(root_data_dir)

