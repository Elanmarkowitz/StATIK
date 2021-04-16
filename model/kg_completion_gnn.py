import torch
from torch import nn
from torch import Tensor

from model.parameterized_sampling import ParameterizedSampler


class MessageCalculationLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(MessageCalculationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transform_message = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, H: Tensor, E: Tensor, heads: Tensor, r_embed: Tensor):
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
        # TODO: Maybe normalize
        return messages


class MessagePassingLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(MessagePassingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.calc_messages_fwd = MessageCalculationLayer(embed_dim)
        self.calc_messages_back = MessageCalculationLayer(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.LeakyReLU()

    def aggregate_messages(self, ht: Tensor, messages_fwd: Tensor, messages_back: Tensor, nodes_in_batch: int,
                           p_selections: Tensor = None):
        """
        :param ht: shape m x 2
        :param messages_fwd: shape m x embed_dim
        :param messages_back: shape m x embed_dim
        :param p_selections: shape m x 1
        :param nodes_in_batch: int
        :return: aggregated messages. shape nodes_in_batch x embed_dim
        """
        device = messages_fwd.device
        msg_dst = torch.cat([ht[:, 1], ht[:, 0]])
        if p_selections is not None:
            messages_fwd = messages_fwd * p_selections / p_selections.detach()
            messages_back = messages_back * p_selections / p_selections.detach()
        messages = torch.cat([messages_fwd, messages_back], dim=0)
        agg_messages = torch.zeros((nodes_in_batch, self.embed_dim), dtype=messages_fwd.dtype, device=device)
        agg_messages = torch.scatter_add(agg_messages, 0, msg_dst.reshape(-1, 1).expand(-1, self.embed_dim), messages)
        num_msgs = torch.zeros((nodes_in_batch,), dtype=torch.float, device=device)
        unique, counts = msg_dst.unique(return_counts=True)
        num_msgs[unique] = counts.float()
        agg_messages = agg_messages / num_msgs.reshape(-1, 1)  # take mean of messages
        return agg_messages

    def forward(self, H: Tensor, E: Tensor, ht: Tensor, r_embed, p_selections: Tensor = None):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param ht: shape m x 2
        :param r_embed: shape m x embed_dim
        :param p_selections: shape m
        :return:
        """
        messages_fwd = self.calc_messages_fwd(H, E, ht[:, 0], r_embed)
        messages_back = self.calc_messages_back(H, E, ht[:, 1], r_embed)
        if p_selections is not None:
            p_selections = p_selections.reshape(-1, 1)
        aggregated_messages = self.aggregate_messages(ht, messages_fwd, messages_back, H.shape[0], p_selections)
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
        self.edge_update = nn.Linear(3*embed_dim, embed_dim)
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
    def __init__(self, num_relations: int, input_dim: int, embed_dim: int, num_layers: int,
                 parameterized_sampling: bool = False):
        super(KGCompletionGNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        if parameterized_sampling:
            self.head_sampler = ParameterizedSampler(num_relations, 2*num_relations)
            self.tail_sampler = ParameterizedSampler(num_relations, 2*num_relations)

        self.relation_embedding = nn.Embedding(num_relations, embed_dim)
        self.edge_input_transform = nn.Linear(input_dim + 1, embed_dim)
        self.entity_input_transform = nn.Linear(input_dim, embed_dim)
        self.norm_entity = nn.LayerNorm(embed_dim)
        self.norm_edge = nn.LayerNorm(embed_dim)

        self.message_passing_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassingLayer(embed_dim))
            self.edge_update_layers.append(EdgeUpdateLayer(embed_dim))

        self.classify_triple = TripleClassificationLayer(embed_dim)

        self.act = nn.LeakyReLU()

    def forward(self, ht: Tensor, r_tensor: Tensor, entity_feat: Tensor, relation_feat: Tensor, p_selections: Tensor,
                queries: Tensor,) -> Tensor:
        """
        :param ht: m x 2
        :param r_tensor: m
        :param entity_feat: n x feat_dim
        :param relation_feat: m x feat_dim
        :param p_selections: m or None (optional)
        :param queries: m
        :return:
        """
        r_embed = self.relation_embedding(r_tensor)
        H_0 = self.act(self.entity_input_transform(entity_feat))
        H_0 = self.norm_entity(H_0)
        H = H_0

        E_0 = relation_feat[r_tensor]
        E_0 = self.act(self.edge_input_transform(torch.cat([E_0, queries.reshape(-1, 1)], dim=1)))
        E_0 = self.norm_edge(E_0)
        E = E_0

        for i in range(self.num_layers):
            H = self.message_passing_layers[i](H, E, ht, r_embed, p_selections)
            E = self.edge_update_layers[i](H, E, ht)

        out = self.classify_triple(H, E, H_0, E_0, ht, queries)
        return out

