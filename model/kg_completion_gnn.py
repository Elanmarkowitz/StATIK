import inspect

import torch
import torch.nn.functional as F
import math
from torch import nn
from torch import Tensor


class MessageCalculationLayer(nn.Module):
    def __init__(self, embed_dim: int, message_weighting_function=None):
        super(MessageCalculationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transform_message = nn.Linear(2 * embed_dim, embed_dim)
        self.message_weighting_function = message_weighting_function

    def forward(self, H: Tensor, E: Tensor, heads: Tensor, queries: Tensor):
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
        message_weights = self.message_weighting_function(E, queries) if self.message_weighting_function is not None else None

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

    @staticmethod
    def aggregate_messages(ht: Tensor, messages_fwd: Tensor, messages_back: Tensor, nodes_in_batch: int):
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
        agg_messages = torch.zeros((nodes_in_batch, messages_fwd.shape[1]), dtype=messages_fwd.dtype, device=device)
        agg_messages = torch.scatter_add(agg_messages, 0, msg_dst.reshape(-1, 1).expand(-1, messages_fwd.shape[1]), messages)
        num_msgs = torch.zeros((nodes_in_batch,), dtype=torch.float, device=device)
        unique, counts = msg_dst.unique(return_counts=True)
        num_msgs[unique] = counts.float()
        agg_messages = agg_messages / num_msgs.reshape(-1, 1)  # take mean of messages

        return agg_messages

    def forward(self, H: Tensor, E: Tensor, ht: Tensor, queries):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param ht: shape m x 2
        :param r_embed: shape m x embed_dim
        :return:
        """
        messages_fwd = self.calc_messages_fwd(H, E, ht[:, 0], queries)
        messages_back = self.calc_messages_back(H, E, ht[:, 1], queries)
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


class RelationCorrelationModel(nn.Module):
    def __init__(self, num_relations: int, embed_dim: int):
        super(RelationCorrelationModel, self).__init__()
        self.relation_correlation_embedding = nn.Embedding(num_relations, embed_dim)
        self.ht_transform = nn.Linear(embed_dim, embed_dim)
        self.hh_transform = nn.Linear(embed_dim, embed_dim)
        self.th_transform = nn.Linear(embed_dim, embed_dim)
        self.tt_transform = nn.Linear(embed_dim, embed_dim)
        self.correlation_weight_table = nn.Parameter(torch.ones(num_relations, num_relations).float())
        self.final_embedding = nn.Linear(2 * embed_dim, embed_dim)
        self.final_score = nn.Linear(embed_dim, 1)

    def forward(self, ht: Tensor, r_q: Tensor, r_tensor: Tensor, r_relative: Tensor, h_or_t_sample: Tensor,
                queries: Tensor, num_nodes: int):
        r_corr_embed = self.relation_correlation_embedding(r_tensor)
        r_corr_embed = r_corr_embed * torch.sigmoid(self.correlation_weight_table[r_tensor, r_q]).view(-1, 1)

        # compute the relation embedding for all topologies
        hh_embed = self.hh_transform(r_corr_embed)
        ht_embed = self.ht_transform(r_corr_embed)
        th_embed = self.th_transform(r_corr_embed)
        tt_embed = self.tt_transform(r_corr_embed)
        all_embed = torch.stack([torch.stack([tt_embed, th_embed]), torch.stack([ht_embed, hh_embed])])

        # select the topology present
        selected_embed = all_embed[r_relative, h_or_t_sample, torch.arange(r_relative.shape[0])]

        # query relationship does not pass information
        selected_embed = selected_embed * torch.logical_not(queries).float().view(-1,1)

        aggregated_to_nodes = MessagePassingLayer.aggregate_messages(ht, selected_embed, selected_embed, num_nodes)

        query_idx = queries.nonzero().flatten()
        query_entities = ht[query_idx]
        query_r_type = r_tensor[query_idx]

        aggregated_to_query = aggregated_to_nodes[query_entities[:,0]] + aggregated_to_nodes[query_entities[:,1]]

        final_query_embedding = self.final_embedding(torch.cat([aggregated_to_query, self.relation_correlation_embedding(query_r_type)], dim=1))

        return self.final_score(final_query_embedding)


class KGCompletionGNN(nn.Module):
    def __init__(self, relation_feat, input_dim: int, embed_dim: int, num_layers: int, norm=2, edge_attention=False,
                 decoder: str = "MLP+TransE"):
        super(KGCompletionGNN, self).__init__()

        local_vals = locals()
        self.instantiation_args = [local_vals[arg] for arg in inspect.signature(KGCompletionGNN).parameters]
        self.arg_signature = list(inspect.signature(KGCompletionGNN).parameters)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.norm = norm

        self.relation_embedding = nn.Embedding(relation_feat.shape[0], relation_feat.shape[1])
        self.relation_embedding.weight = nn.Parameter(torch.tensor(relation_feat, dtype=torch.float))
        self.relation_embedding.weight.requires_grad = False  # Comment to learn through message passing relation embedding table

        self.edge_input_transform = nn.Linear(relation_feat.shape[1], embed_dim)

        self.entity_input_transform = nn.Linear(input_dim, embed_dim)

        self.message_weighting_function = MessageWeightingFunction(embed_dim, embed_dim // 2) if edge_attention else None
        self.norm_entity = nn.LayerNorm(embed_dim)
        self.norm_edge = nn.LayerNorm(embed_dim)

        self.message_passing_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassingLayer(embed_dim, message_weighting_function=self.message_weighting_function))
            self.edge_update_layers.append(EdgeUpdateLayer(embed_dim))

        self.relation_correlation_model = RelationCorrelationModel(relation_feat.shape[0], embed_dim)

        self.decoder = decoder
        self.classify_triple = TripleClassificationLayer(embed_dim)
        self.transE_decoder = TransEDecoder(relation_feat.shape[0], embed_dim)

        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, ht: Tensor, r_tensor: Tensor, r_query: Tensor, entity_feat: Tensor, r_relative, h_or_t_sample, queries: Tensor):
        # Transform entities

        H_0 = self.act(self.entity_input_transform(entity_feat))
        H_0 = self.norm_entity(H_0)
        H = H_0

        # Transform relations
        r_embed = self.relation_embedding(r_tensor)
        E_0 = self.act(self.edge_input_transform(r_embed))
        E_0 = self.norm_edge(E_0)
        E = E_0

        for i in range(self.num_layers):
            H = self.message_passing_layers[i](H, E, ht, queries)
            E = self.edge_update_layers[i](H, E, ht)

        if self.decoder == "MLP":
            out = self.classify_triple(H, E, H_0, E_0, ht, queries).flatten()
        elif self.decoder == "TransE":
            out = -1 * self.transE_decoder(H, r_tensor, ht, queries)
        elif self.decoder == "MLP+TransE":
            if self.training:
                mlp_out = self.classify_triple(H, E, H_0, E_0, ht, queries).flatten()
                transe_out = -1 * self.transE_decoder(H, r_tensor, ht, queries)
                out = (mlp_out.flatten(), transe_out)
            else:
                out = -1 * self.transE_decoder(H, r_tensor, ht, queries)
        elif self.decoder == "RelCorr+TransE":
            rel_corr_score = self.relation_correlation_model(ht, r_query, r_tensor, r_relative, h_or_t_sample, queries,
                                                             entity_feat.shape[0])
            transe_out = -1 * self.transE_decoder(H, r_tensor, ht, queries)
            out = rel_corr_score.flatten() + transe_out
        elif self.decoder == "RelCorr+MLP":
            rel_corr_score = self.relation_correlation_model(ht, r_query, r_tensor, r_relative, h_or_t_sample, queries,
                                                             entity_feat.shape[0])
            out = rel_corr_score.flatten() + self.classify_triple(H, E, H_0, E_0, ht, queries).flatten()
        else:
            out = None
            Exception('Decoder not valid.')

        return out

    def get_loss_fn(self, margin=1.0):
        if self.decoder == "MLP":
            return nn.BCEWithLogitsLoss()
        elif self.decoder == "TransE":
            self.margin = margin
            return self.margin_ranking_loss
        elif self.decoder == "MLP+TransE":
            self.margin = margin
            return self.combo_loss
        elif self.decoder == "RelCorr+TransE":
            self.margin = margin
            return self.margin_ranking_loss
        elif self.decoder == "RelCorr+MLP":
            self.margin = margin
            return self.margin_ranking_loss
        else:
            Exception(f"Loss function not known for {self.decoder}")

    def margin_ranking_loss(self, scores, labels):
        distances = -1 * scores
        positives = labels.nonzero(as_tuple=False).flatten()
        negatives = (labels == 0).nonzero(as_tuple=False).flatten()
        pos_distances = distances[positives]
        neg_distances = distances[negatives]
        target = torch.tensor([-1], dtype=torch.long, device=scores.device)
        return F.margin_ranking_loss(pos_distances, neg_distances, target, margin=self.margin)

    def combo_loss(self, scores, labels):
        mlp_scores = scores[0]
        transe_scores = scores[1]
        bce_loss = F.binary_cross_entropy_with_logits(mlp_scores, labels)
        transe_loss = self.margin_ranking_loss(transe_scores, labels)
        return transe_loss + 0.4 * bce_loss


class TransEDecoder(nn.Module):
    def __init__(self, num_relations: int, embed_dim: int, distance_norm: int = 2):
        super(TransEDecoder, self).__init__()
        self.distance_norm = distance_norm
        self.relation_vector = nn.Embedding(num_relations, embed_dim)
        self.relation_vector.weight.data.uniform_(-6 / math.sqrt(embed_dim), 6 / math.sqrt(embed_dim))

    def forward(self, H, r_tensor, ht, queries):
        query_idxs = queries.nonzero().flatten()
        ht_q = ht[query_idxs]
        r_q = r_tensor[query_idxs]

        head_embeds = H[ht_q[:, 0]]
        relation_embeds = self.relation_vector(r_q)
        tail_embeds = H[ht_q[:, 1]]

        distances = self.distance(head_embeds, relation_embeds, tail_embeds, self.distance_norm)

        return distances

    @staticmethod
    def distance(head_embeds, relation_embeds, tail_embeds, p):
        head_embeds = F.normalize(head_embeds, p=2, dim=1)
        tail_embeds = F.normalize(tail_embeds, p=2, dim=1)
        distances = torch.norm(head_embeds + relation_embeds - tail_embeds, p=p, dim=1)
        return distances
