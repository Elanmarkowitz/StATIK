from array import array

import torch
import numpy as np
from transformers import BertTokenizer

from data.data_classes import KGGraph, KGLoadableDataset, KGInferenceDataset


class MessagePassingLoadingFunction:
    def __init__(self, graph: KGGraph, max_neighbors=10):
        self.graph = graph
        self.max_neighbors = max_neighbors

    def sample_neighbors(self, node, max_neighbors: int):
        edge_lccsr = self.graph.edge_lccsr
        relation_lccsr = self.graph.relation_lccsr
        if self.graph.degrees[node] > max_neighbors:
            selection = np.random.randint(self.graph.degrees[node], size=(max_neighbors,))
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
        inverse_relation_idx = rels >= self.graph.num_relations
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
        edge_relations.extend(rels[inverse_relation_idx] - self.graph.num_relations)

        return forward_relation_idx

    def create_component(self, entity, rels, tails):
        entity_set = set()
        edge_heads = array("i")
        edge_tails = array("i")
        edge_relations = array("i")

        r_relative = self.add_relations_with_inverting(entity, rels, tails, entity_set, edge_heads, edge_tails, edge_relations)

        entity_set_list = list(entity_set)
        batch_id_to_node_id = np.array(entity_set_list)
        node_id_to_batch_node_id = dict((e, i) for (i, e) in enumerate(entity_set_list))
        edge_heads = np.array([node_id_to_batch_node_id[e] for e in edge_heads])
        edge_tails = np.array([node_id_to_batch_node_id[e] for e in edge_tails])
        query_node = (batch_id_to_node_id == entity).astype(np.int32)

        return (edge_heads, edge_relations, edge_tails, batch_id_to_node_id, query_node, r_relative), len(entity_set)

    @staticmethod
    def add_component(edge_heads, edge_relations, edge_tails, cumulative_entities, batch_id_to_node_id, query_nodes, r_relatives, component):
        c_edge_heads, c_edge_relations, c_edge_tails, c_batch_id_to_node_id, query_node, r_relative = component

        edge_heads.extend(c_edge_heads + cumulative_entities)
        edge_tails.extend(c_edge_tails + cumulative_entities)
        edge_relations.extend(c_edge_relations)
        batch_id_to_node_id.extend(c_batch_id_to_node_id)
        query_nodes.extend(query_node)
        r_relatives.extend(r_relative)
        return

    def __call__(self, batch):
        """
        Messagae Passing Loading Function, takes a list of entities
        :param batch: List[tuple(entity, ignore_r, ignore_t)]
        :return: ht_tensor, r_tensor, r_relative, entity_set
        """
        # query edge marked as query
        # 1-hop connected entities included
        batch_id_to_node_id = array("i")
        edge_heads = array("i")
        edge_tails = array("i")
        edge_relations = array("i")
        query_nodes = array("b")
        r_relatives = array("i")
        cumulative_entities = 0

        for _entity, _ignore_r, _ignore_t in batch:
            sampled_rels, sampled_tails = self.sample_neighbors(_entity, self.max_neighbors)
            sampled_rels, sampled_tails = self.ignore_query(sampled_rels, sampled_tails, _ignore_r, _ignore_t)
            component, c_size = self.create_component(_entity, sampled_rels, sampled_tails)
            self.add_component(edge_heads, edge_relations, edge_tails, cumulative_entities, batch_id_to_node_id,
                               query_nodes, r_relatives, component)
            cumulative_entities += c_size

        ht_tensor = torch.from_numpy(np.stack([edge_heads, edge_tails]).transpose()).long()
        r_tensor = torch.from_numpy(np.array(edge_relations)).long()
        entity_set = torch.from_numpy(np.array(batch_id_to_node_id)).long()
        query_nodes = torch.from_numpy(np.array(query_nodes)).bool()
        r_relatives = torch.from_numpy(np.array(r_relatives)).long()
        return ht_tensor, r_tensor, entity_set, query_nodes, r_relatives


class TokenizerLoadingFunction:
    def __init__(self, tokenizer: BertTokenizer, entity_text, relation_text):
        self.tokenizer = tokenizer
        self.entity_text = entity_text
        self.relation_text = relation_text

    def __call__(self, batch):
        """
        :param batch: a list of (head, relation, tail, head_prediction: bool, is_query: bool)
        :return: tokenized results
        """
        texts = []
        for h, r, t, head_prediction, is_query in batch:
            text = self._get_text(h, r, t, head_prediction, is_query)
            texts.append(text)
        res = self.tokenizer(texts, padding=True, return_tensors='pt')
        input_ids = res['input_ids']
        token_type_ids = res['token_type_ids']
        attention_mask = res['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def _get_text(self, h: int, r: int, t: int, head_prediction: bool, is_query: bool):
        if is_query:
            entity_text = self._get_entity_text(t) if head_prediction else self._get_entity_text(h)
            relation_text = self._get_relation_text(r)
            return self._combine_text(entity_text, relation_text, head_prediction=head_prediction)
        else:  # is_target
            entity_text = self._get_entity_text(h) if head_prediction else self._get_entity_text(t)
            return self._combine_text(entity_text)

    def _get_entity_text(self, entity):
        return self.entity_text[entity]

    def _get_relation_text(self, relation):
        return self.relation_text[relation]

    def _combine_text(self, entity_text, relation_text=None, head_prediction=False):
        text = self._get_n_words(entity_text)
        if relation_text is not None:
            relation_text = f"inverse of {relation_text}" if head_prediction else relation_text
            text = text + self.tokenizer.sep_token + self._get_n_words(relation_text)

        return text  # TODO (optional): deal with token_type_ids

    @staticmethod
    def _get_n_words(text, n=32):
        return ' '.join(text.split(' ')[:n])


class TrainingCollateFunction:
    def __init__(self, dataset: KGLoadableDataset, max_neighbors, num_negatives):
        self.tokenizer_loading_fn = TokenizerLoadingFunction(BertTokenizer.from_pretrained('bert-base-cased'),
                                                             entity_text=dataset.base_dataset.entity_text,
                                                             relation_text=dataset.base_dataset.relation_text)
        self.message_passing_loading_fn = MessagePassingLoadingFunction(dataset.graph, max_neighbors=max_neighbors)
        self.num_negatives = num_negatives
        self.num_relations = dataset.base_dataset.num_relations
        self.training_entities = dataset.graph.present_entities

    def __call__(self, batch):
        # list of hrt
        negatives = self._get_negatives(self.num_negatives)
        token_queries = np.concatenate([np.ones((2 * len(batch),)),
                                        np.zeros((2 * len(batch),)),
                                        np.zeros((len(negatives),))])
        positive_targets = np.concatenate([np.zeros((2 * len(batch),)),
                                           np.ones((2 * len(batch),)),
                                           np.zeros((len(negatives),))])
        negative_targets = np.concatenate([np.zeros((2 * len(batch),)),
                                           np.zeros((2 * len(batch),)),
                                           np.ones((len(negatives),))])
        is_head_prediction = np.concatenate([np.zeros((len(batch),)),
                                             np.ones((len(batch),))])

        to_tokenizer = self._prepare_for_tokenizer_fn(batch, negatives)
        input_ids, token_type_ids, attention_mask = self.tokenizer_loading_fn(to_tokenizer)
        to_mp_loader = self._prepare_for_mp_loading_fn(batch, negatives)
        ht_tensor, r_tensor, entity_set, query_nodes, r_relatives = self.message_passing_loading_fn(to_mp_loader)

        r_query = torch.tensor(2*[r for h, r, t in batch]).long()

        token_queries = torch.from_numpy(token_queries).bool()
        positive_targets = torch.from_numpy(positive_targets).bool()
        negative_targets = torch.from_numpy(negative_targets).bool()
        is_head_prediction = torch.from_numpy(is_head_prediction).bool()

        entity_feat = None

        return (input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction), \
            token_queries, positive_targets, negative_targets

    def _get_negatives(self, num_negatives):
        return self.training_entities[np.random.randint(0, len(self.training_entities), (num_negatives,))]

    def _prepare_for_mp_loading_fn(self, batch, negatives=None):
        for_head = [(h, r, t) for h, r, t in batch]
        for_tail = [(t, r + self.num_relations, h) for h, r, t in batch]

        # query-tail pred, query-head pred, target-tail pred, target-head_pred
        to_mp_loader = for_head + for_tail + for_tail + for_head

        if negatives is not None:
            to_mp_loader += [(n, -1, -1) for n in negatives]

        return to_mp_loader

    @staticmethod
    def _prepare_for_tokenizer_fn(batch, negatives=None):
        to_tokenizer = []

        # Add queries
        to_tokenizer += [(h, r, t, False, True) for h, r, t in batch]  # tail prediction
        to_tokenizer += [(h, r, t, True, True) for h, r, t in batch]  # head prediction

        # Add positive targets
        to_tokenizer += [(h, r, t, False, False) for h, r, t in batch]  # tail prediction
        to_tokenizer += [(h, r, t, True, False) for h, r, t in batch]  # head prediction

        if negatives is not None:
            to_tokenizer += [(-1, -1, n, False, False) for n in negatives]

        return to_tokenizer


class InferenceCollateTargetFunction:
    def __init__(self, dataset: KGInferenceDataset, max_neighbors):
        self.tokenizer_loading_fn = TokenizerLoadingFunction(BertTokenizer.from_pretrained('bert-base-cased'),
                                                             entity_text=dataset.base_dataset.entity_text,
                                                             relation_text=dataset.base_dataset.relation_text)
        self.message_passing_loading_fn = MessagePassingLoadingFunction(dataset.graph, max_neighbors=max_neighbors)

    def __call__(self, batch):
        to_tokenizer = [(-1, -1, target, False, False) for target in batch]
        to_mp_loader = [(target, -1, -1) for target in batch]
        input_ids, token_type_ids, attention_mask = self.tokenizer_loading_fn(to_tokenizer)
        ht_tensor, r_tensor, entity_set, query_nodes, r_relatives = self.message_passing_loading_fn(to_mp_loader)
        entity_feat = None

        # No queries so empty
        r_query = torch.tensor([]).long()
        is_head_prediction = torch.tensor([]).bool()

        return input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction


class InferenceCollateQueryFunction:
    def __init__(self, dataset: KGInferenceDataset, max_neighbors, head_prediction: bool):
        self.tokenizer_loading_fn = TokenizerLoadingFunction(BertTokenizer.from_pretrained('bert-base-cased'),
                                                             entity_text=dataset.base_dataset.entity_text,
                                                             relation_text=dataset.base_dataset.relation_text)
        self.message_passing_loading_fn = MessagePassingLoadingFunction(dataset.graph, max_neighbors=max_neighbors)
        self.num_relations = dataset.base_dataset.num_relations
        self.head_prediction = head_prediction

    def __call__(self, batch):
        if self.head_prediction:
            filter_mask = [mask for _, mask, _ in batch]
        else:
            filter_mask = [mask for _, _, mask, in batch]
        filter_mask = torch.from_numpy(np.stack(filter_mask, axis=0))
        to_tokenizer = self._prepare_for_tokenizer(batch)
        to_mp_loader = self._prepare_for_mp_loader(batch)
        input_ids, token_type_ids, attention_mask = self.tokenizer_loading_fn(to_tokenizer)
        ht_tensor, r_tensor, entity_set, query_nodes, r_relatives = self.message_passing_loading_fn(to_mp_loader)

        true_targets = np.concatenate([np.zeros((len(batch),)),
                                       np.ones((len(batch),))])
        true_targets = torch.from_numpy(true_targets).bool()

        entity_feat = None

        r_query = torch.tensor([r for (_, r, _), _, _ in batch])
        is_head_prediction = torch.tensor([self.head_prediction]).repeat(len(batch))

        return (input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction), \
            true_targets, filter_mask

    def _prepare_for_tokenizer(self, batch):
        to_tokenizer = []
        for (h, r, t), _, _, in batch:
            to_tokenizer.append((h, r, t, self.head_prediction, True))  # add query
        for (h, r, t), _, _, in batch:
            to_tokenizer.append((h, r, t, self.head_prediction, False))  # add target
        return to_tokenizer

    def _prepare_for_mp_loader(self, batch):
        if self.head_prediction:
            for_query = [(t, r + self.num_relations, h) for (h, r, t), _, _ in batch]
            for_target = [(h, r, t) for (h, r, t), _, _ in batch]
        else:
            for_query = [(h, r, t) for (h, r, t), _, _ in batch]
            for_target = [(t, r + self.num_relations, h) for (h, r, t), _, _ in batch]
        to_mp = for_query + for_target
        return to_mp
