# Branch keshav

import torch
from torch import nn
from torch import Tensor

import pdb


class MessageCalculationLayer(nn.Module):
    def __init__(self, embed_dim: int, edge_attention=False):
        super(MessageCalculationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transform_message = nn.Linear(4 * embed_dim, embed_dim)
        self.message_weighting_function = MessageWeightingFunction(embed_dim, embed_dim // 2)
        self.edge_attention = edge_attention

    def forward(self, H: Tensor, E: Tensor, heads: Tensor, r_embed: Tensor, queries: Tensor):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param heads: shape m x 1
        :param r_embed: shape m x embed_dim
        :return:
            processed messages for nodes. shape nodes_in_batch x embed_dim
        """
        H_heads = H[heads]
        raw_messages = torch.cat([H_heads, E, H_heads * r_embed, E * r_embed], dim=1)
        messages = self.transform_message(raw_messages)
        message_weights = self.message_weighting_function(E, queries) if self.edge_attention else None

        # TODO: Maybe normalize
        return message_weights * messages if self.edge_attention else messages


class MessageWeightingFunction(nn.Module):
    def __init__(self, relation_embed_dim: int, attention_dim: int):
        super(MessageWeightingFunction, self).__init__()
        self.relation_embed_dim = relation_embed_dim
        self.attention_dim = attention_dim
        self.Q = nn.Linear(relation_embed_dim, attention_dim, bias=False)
        self.K = nn.Linear(relation_embed_dim, attention_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)

    # Computes the attention scores between the relation embeddings of all sampled edges with the relation type of the query
    def compute_attention_scores(self, Q: Tensor, K: Tensor):
        attention_scores = torch.matmul(Q, K.T)
        return self.softmax(attention_scores)

    def forward(self, relation_embeds: Tensor, queries: Tensor):
        attention_scores = []
        batch_separation_points = [0] + (queries.nonzero(as_tuple=False).flatten() + 1).tolist()
        for i in range(1, len(batch_separation_points)):
            batch_slice = relation_embeds[batch_separation_points[i - 1]: batch_separation_points[i], :]
            Q = self.Q(batch_slice[-1, :])  # Transformed Q of the query relation
            K = self.K(batch_slice)  # Transformed Ks of all the sampled edges + query edge
            batch_attention_scores = self.compute_attention_scores(Q, K)
            attention_scores.append(batch_attention_scores)

        return torch.cat(attention_scores)


class MessagePassingLayer(nn.Module):
    def __init__(self, embed_dim: int, edge_attention=False):
        super(MessagePassingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.calc_messages_fwd = MessageCalculationLayer(embed_dim, edge_attention)
        self.calc_messages_back = MessageCalculationLayer(embed_dim, edge_attention)
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

    def forward(self, H: Tensor, E: Tensor, ht: Tensor, r_embed, queries):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param ht: shape m x 2
        :param r_embed: shape m x embed_dim
        :return:
        """
        messages_fwd = self.calc_messages_fwd(H, E, ht[:, 0], r_embed, queries)
        messages_back = self.calc_messages_back(H, E, ht[:, 1], r_embed, queries)
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
    def __init__(self, num_relations: int, input_dim: int, embed_dim: int, num_layers: int, edge_attention=False):
        super(KGCompletionGNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.relation_embedding = nn.Embedding(num_relations, embed_dim)
        self.edge_input_transform = nn.Linear(input_dim + 1, embed_dim)
        self.entity_input_transform = nn.Linear(input_dim, embed_dim)
        self.norm_entity = nn.LayerNorm(embed_dim)
        self.norm_edge = nn.LayerNorm(embed_dim)

        self.message_passing_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassingLayer(embed_dim, edge_attention=edge_attention))
            self.edge_update_layers.append(EdgeUpdateLayer(embed_dim))

        self.classify_triple = TripleClassificationLayer(embed_dim)

        self.act = nn.LeakyReLU()

    def forward(self, ht: Tensor, r_tensor: Tensor, entity_feat: Tensor, relation_feat: Tensor, queries: Tensor) -> Tensor:
        r_embed = self.relation_embedding(r_tensor)
        H_0 = self.act(self.entity_input_transform(entity_feat))
        H_0 = self.norm_entity(H_0)
        H = H_0

        E_0 = relation_feat[r_tensor]
        E_0 = self.act(self.edge_input_transform(torch.cat([E_0, queries.reshape(-1, 1)], dim=1)))
        E_0 = self.norm_edge(E_0)
        E = E_0

        for i in range(self.num_layers):
            H = self.message_passing_layers[i](H, E, ht, r_embed, queries)
            E = self.edge_update_layers[i](H, E, ht)

        out = self.classify_triple(H, E, H_0, E_0, ht, queries)

        return out
