from ogb.lsc import WikiKG90MDataset
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np 

from data import data_processing

class WikiKG90MProcessedDataset(Dataset):
    """WikiKG90M processed dataset."""

    def __init__(self, root_data_dir: str, include_inverse: bool=True, from_dataset=None):
        """
        Args:
            root_data_dir (str): Root data dir containing the processed WikiKG90M dataset.
        """
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
        self.entity_feat = dataset.entity_feat # feature matrix for entities
        self.relation_feat = dataset.relation_feat # feature matrix for relation types
        self.edge_dict = dataset.edge_dict # dict from head id to list of tail ids #includes both pairs at the moment
        self.relation_dict = dataset.relation_dict # corresponding to edge_dict, dict from head id to list of relation ids #includes both directions at the moment
        self.degrees = dataset.degrees

    @staticmethod
    def init_post_processing(processed_dataset):
        """Load directly from results of data_processing.load_processed_data
        Primarily used for development purposes.
        """
        return WikiKG90MProcessedDataset(None, from_dataset=processed_dataset)

    def __len__(self):
        return len(self.train_ht)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_ht = self.train_ht[idx]
        batch_r = self.train_r[idx]

        return batch_ht, batch_r

    def get_collate_fn(self, single_query_per_head=True, max_neighbors=100):
        def wikikg_collate_fn(batch):
            # query edge marked as query
            # 1-hop connected entities included

            entity_set = set()
            edge_heads = []
            edge_tails = []
            edge_relations = []
            is_query = []
            labels = []
            for _ht, _r in batch:
                # get neighbors of h and t
                _h, _t = _ht[0], _ht[1]
                _label = 1
                
                if single_query_per_head:
                    if random.choice([True, False]):
                        _t = random.randrange(self.num_entities)
                        _label = 0

                if self.degrees[_h] > max_neighbors:
                    selection = np.random.randint(self.degrees[_h], size=(max_neighbors,))
                else: 
                    selection = np.arange(self.degrees[_h])
                for tail, rel in zip(self.edge_dict[_h][selection], self.relation_dict[_h][selection]):
                    if tail == _t and rel == _r: # do not include query relation here
                        continue
                    entity_set.add(tail)
                    edge_heads.append(_h)
                    edge_tails.append(tail)
                    edge_relations.append(rel)
                    is_query.append(0)
                if self.degrees[_t] > max_neighbors:
                    selection = np.random.randint(self.degrees[_t], size=(max_neighbors,))
                else:
                    selection = np.arange(self.degrees[_t])
                for tail, rel in zip(self.edge_dict[_t][selection], self.relation_dict[_t][selection]):
                    if self.includes_inverse and tail == _h and rel == _r + self.num_entities/2: # do not include inverse of query relation here
                        continue
                    entity_set.add(tail)
                    edge_heads.append(_t)
                    edge_tails.append(tail)
                    edge_relations.append(rel)
                    is_query.append(0)

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
                    if self.degrees[neg_t] > max_neighbors:
                        selection = np.random.randint(self.degrees[_t], size=(max_neighbors,))
                    else:
                        selection = np.arange(self.degrees[_t])
                    for tail, rel in zip(self.edge_dict[neg_t], self.relation_dict[neg_t]):
                        entity_set.add(tail)
                        edge_heads.append(neg_t)
                        edge_tails.append(tail)
                        edge_relations.append(rel)
                        is_query.append(0)
                    entity_set.add(neg_t)
                    edge_heads.append(_h)
                    edge_tails.append(neg_t)
                    edge_relations.append(_r)
                    is_query.append(1)
                    labels.append(0)

            ht_tensor = torch.LongTensor([edge_heads, edge_tails])
            r_tensor = torch.LongTensor(edge_relations)
            entity_set = torch.LongTensor(list(entity_set))
            queries = torch.LongTensor(is_query)
            labels = torch.LongTensor(labels)
            return ht_tensor, r_tensor, entity_set, queries, labels
        return wikikg_collate_fn


def load_dataset(root_data_dir):
    return WikiKG90MProcessedDataset(root_data_dir)
