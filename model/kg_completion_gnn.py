# Branch keshav

import torch
import torch.nn.functional as F
import math
from torch import nn
from torch import Tensor

import pdb


class MessageCalculationLayer(nn.Module):
    def __init__(self, embed_dim: int, message_weighting_function=None):
        super(MessageCalculationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transform_message = nn.Linear(2 * embed_dim, embed_dim)
        self.message_weighting_function = message_weighting_function

    def forward(self, H: Tensor, E: Tensor, r_embed: Tensor, heads: Tensor, queries: Tensor):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param heads: shape m x 1
        :param r_embed: shape m x embed_dim
        :return:
            processed messages for nodes. shape nodes_in_batch x embed_dim
        """
        H_heads = H[heads]
        raw_messages = torch.cat([H_heads, E], dim=1)
        messages = self.transform_message(raw_messages)
        message_weights = self.message_weighting_function(r_embed, queries) if self.message_weighting_function is not None else None

        # TODO: Maybe normalize
        return message_weights.view(-1, 1) * messages if self.message_weighting_function is not None else messages


class MessageWeightingFunction(nn.Module):
    def __init__(self, relation_embed_dim: int, attention_dim: int):
        super(MessageWeightingFunction, self).__init__()
        self.relation_embed_dim = relation_embed_dim
        self.attention_dim = attention_dim
        self.Q = nn.Linear(relation_embed_dim, attention_dim, bias=False)
        self.K = nn.Linear(relation_embed_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, relation_embeds: Tensor, queries: Tensor):
        query_idxs = queries.nonzero(as_tuple=False).flatten()
        value_idxs = torch.roll(queries, shifts=1)
        value_idxs[0] = 0
        X_attn_idxs = torch.cumsum(value_idxs, dim=0)
        Y_attn_idxs = torch.arange(relation_embeds.shape[0])

        Q = self.Q(relation_embeds)
        K = self.K(relation_embeds)
        Q_query = Q[query_idxs]
        attn_matrix = torch.matmul(K, Q_query.T) / math.sqrt(self.attention_dim)

        negative_y, negative_x = torch.where(attn_matrix < 0)
        mask = torch.full(attn_matrix.shape, -1e+30, device=attn_matrix.device)
        mask[negative_y, negative_x] = 1e+30
        mask[Y_attn_idxs, X_attn_idxs] = 1

        attn_scores = attn_matrix * mask.detach()
        attn_scores = self.softmax(attn_scores)
        attn_scores = attn_scores[Y_attn_idxs, X_attn_idxs]

        return attn_scores


class MessagePassingLayer(nn.Module):
    def __init__(self, embed_dim: int, message_weighting_function=None):
        super(MessagePassingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.calc_messages_fwd = MessageCalculationLayer(embed_dim, message_weighting_function)
        self.calc_messages_back = MessageCalculationLayer(embed_dim, message_weighting_function)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.LeakyReLU()

    def aggregate_messages(self, ht: Tensor, messages_fwd: Tensor, messages_back: Tensor, nodes_in_batch: int):
        """
        :param ht: shape m x 2
        :param messages_fwd: shape m x embed_dim
        :param messages_back: shape m x embed_dim
        :param nodes_in_batch: int
        :return: aggregated messages. shape nodes_in_batch x embed_dim
        """
        device = messages_fwd.device
        msg_dst = torch.cat([ht[:, 1], ht[:, 0]])
        messages = torch.cat([messages_fwd, messages_back], dim=0)
        agg_messages = torch.zeros((nodes_in_batch, self.embed_dim), dtype=messages_fwd.dtype, device=device)
        agg_messages = torch.scatter_add(agg_messages, 0, msg_dst.reshape(-1, 1).expand(-1, self.embed_dim), messages)
        num_msgs = torch.zeros((nodes_in_batch,), dtype=torch.float, device=device)
        unique, counts = msg_dst.unique(return_counts=True)
        num_msgs[unique] = counts.float()
        agg_messages = agg_messages / num_msgs.reshape(-1, 1)  # take mean of messages

        return agg_messages

    def forward(self, H: Tensor, E: Tensor, r_embed: Tensor, ht: Tensor, queries):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param ht: shape m x 2
        :param r_embed: shape m x embed_dim
        :return:
        """
        messages_fwd = self.calc_messages_fwd(H, E, r_embed, ht[:, 0], queries)
        messages_back = self.calc_messages_back(H, E, r_embed, ht[:, 1], queries)
        aggregated_messages = self.aggregate_messages(ht, messages_fwd, messages_back, H.shape[0])
        out = self.norm(self.act(aggregated_messages) + H)
        return out


class TripleClassificationLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(TripleClassificationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.layer1 = nn.Linear(6 * embed_dim, embed_dim)
        self.layer2 = nn.Linear(embed_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, H: Tensor, E: Tensor, H_0: Tensor, E_0: Tensor, ht: Tensor, queries: Tensor):
        query_indices = queries.nonzero().flatten()
        E_q = E[query_indices]
        E_0_q = E_0[query_indices]
        ht_q = ht[query_indices]
        H_head_q = H[ht_q[:, 0]]
        H_tail_q = H[ht_q[:, 1]]
        H_0_head_q = H_0[ht_q[:, 0]]
        H_0_tail_q = H_0[ht_q[:, 1]]
        out_1 = self.act(self.layer1(
            torch.cat([E_q, E_0_q, H_head_q, H_0_head_q, H_tail_q, H_0_tail_q], dim=1)
        ))
        out = self.layer2(out_1)
        return out


class EdgeUpdateLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(EdgeUpdateLayer, self).__init__()
        self.embed_dim = embed_dim
        self.alpha = nn.Parameter(torch.tensor(0.0).float())
        self.edge_update = nn.Linear(3 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.LeakyReLU()

    def forward(self, H: Tensor, E: Tensor, ht: Tensor):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param ht: shape m x 2
        :return:
        """
        neighbor_representations = H[ht]  # shape m x 2 x embed_dim
        head_rep = neighbor_representations[:, 0]  # m x embed_dim
        tail_rep = neighbor_representations[:, 1]  # m x embed_dim
        triple_rep = torch.cat([head_rep, E, tail_rep], dim=1)  # m x 3*embed_dim
        edge_update = self.act(self.edge_update(triple_rep))
        out = self.norm(edge_update + E)
        return out


class KGCompletionGNN(nn.Module):
    def __init__(self, relation_feat, input_dim: int, embed_dim: int, num_layers: int, norm=2, edge_attention=False):
        super(KGCompletionGNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.norm = norm

        self.relation_embedding = nn.Embedding(relation_feat.shape[0], relation_feat.shape[1])
        self.relation_embedding.weight = nn.Parameter(torch.tensor(relation_feat, dtype=torch.float))
        self.relation_embedding.weight.requires_grad = False  # Comment to learn through message passing relation embedding table

        self.edge_input_transform = nn.Linear(relation_feat.shape[1], embed_dim)
        self.transE_relation_embedding = nn.Embedding(relation_feat.shape[0], embed_dim)
        self.transE_relation_embedding.weight.data.uniform_(-6 / math.sqrt(embed_dim), 6 / math.sqrt(embed_dim))

        self.entity_input_transform = nn.Linear(input_dim, embed_dim)

        self.message_weighting_function = MessageWeightingFunction(relation_feat.shape[1], embed_dim) if edge_attention else None
        self.norm_entity = nn.LayerNorm(embed_dim)
        self.norm_edge = nn.LayerNorm(embed_dim)

        self.message_passing_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassingLayer(embed_dim, message_weighting_function=self.message_weighting_function))
            self.edge_update_layers.append(EdgeUpdateLayer(embed_dim))

        self.classify_triple = TripleClassificationLayer(embed_dim)

        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, ht: Tensor, r_tensor: Tensor, entity_feat: Tensor, queries: Tensor):
        # Transform entities
        H_0 = self.act(self.entity_input_transform(entity_feat))
        H_0 = self.norm_entity(H_0)
        H = H_0

        # Transform relations
        r_embed = self.relation_embedding(r_tensor)
        E_0 = self.act(self.edge_input_transform(r_embed))
        E_0 = self.norm_edge(E_0)
        E = E_0
        E_transE = self.transE_relation_embedding(r_tensor)

        for i in range(self.num_layers):
            H = self.message_passing_layers[i](H, E, r_embed, ht, queries)
            E = self.edge_update_layers[i](H, E, ht)

        out = self.classify_triple(H, E, H_0, E_0, ht, queries)

        if not self.training:
            query_idxs = queries.nonzero().flatten()
            batch_negative_distances = -1 * TransELoss.distance(H[ht[query_idxs, 0]], E_transE[query_idxs], H[ht[query_idxs, 1]], self.norm)
            return batch_negative_distances
        else:
            # Return the updated entity embeddings and relation embeddings as well as distanced for the entire batch
            return H, E_transE, out


class TransELoss(nn.Module):
    def __init__(self, margin, distance_norm=2):
        super(TransELoss, self).__init__()
        self.margin = margin
        self.distance_norm = distance_norm
        self.criterion = nn.MarginRankingLoss(margin=margin)

    def forward(self, H, E, ht, labels, queries, y):
        positives = labels.nonzero(as_tuple=False).flatten()
        negatives = (labels == 0).nonzero(as_tuple=False).flatten()
        query_idxs = queries.nonzero().flatten()
        ht_q = ht[query_idxs]
        E_q = E[query_idxs]

        positive_head_embeds = H[ht_q[positives, 0]]
        positive_tail_embeds = H[ht_q[positives, 1]]
        positive_relation_embeds = E_q[positives]

        negative_head_embeds = H[ht_q[negatives, 0]]
        negative_tail_embeds = H[ht_q[negatives, 1]]
        negative_relation_embeds = E_q[negatives]

        positive_distances = TransELoss.distance(positive_head_embeds, positive_relation_embeds, positive_tail_embeds, self.distance_norm)
        negative_distances = TransELoss.distance(negative_head_embeds, negative_relation_embeds, negative_tail_embeds, self.distance_norm)

        return self.criterion(positive_distances, negative_distances, y)

    @staticmethod
    def distance(head_embeds, relation_embeds, tail_embeds, p):
        head_embeds = F.normalize(head_embeds, p=2, dim=1)
        tail_embeds = F.normalize(tail_embeds, p=2, dim=1)
        distances = torch.norm(head_embeds + relation_embeds - tail_embeds, p=p, dim=1)
        return distances
