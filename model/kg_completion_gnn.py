import inspect

import torch
import torch.nn.functional as F
import math
from torch import nn
from torch import Tensor
from transformers import BertModel


class MessageCalculationLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(MessageCalculationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transform_message = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, H: Tensor, E: Tensor, heads: Tensor):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param heads: shape m x 1
        :return:
            processed messages for nodes. shape nodes_in_batch x embed_dim
        """
        H_heads = H[heads]
        raw_messages = torch.cat([H_heads, E], dim=1)
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
        agg_messages = agg_messages / (num_msgs.reshape(-1, 1) + 1e-7)  # take mean of messages

        return agg_messages

    def forward(self, H: Tensor, E: Tensor, ht: Tensor):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param ht: shape m x 2
        :return:
        """
        messages_fwd = self.calc_messages_fwd(H, E, ht[:, 0])
        messages_back = self.calc_messages_back(H, E, ht[:, 1])
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
        selected_embed = selected_embed * torch.logical_not(queries).float().view(-1, 1)

        aggregated_to_nodes = MessagePassingLayer.aggregate_messages(ht, selected_embed, selected_embed, num_nodes)

        query_idx = queries.nonzero().flatten()
        query_entities = ht[query_idx]
        query_r_type = r_tensor[query_idx]

        aggregated_to_query = aggregated_to_nodes[query_entities[:, 0]] + aggregated_to_nodes[query_entities[:, 1]]

        final_query_embedding = self.final_embedding(torch.cat([aggregated_to_query, self.relation_correlation_embedding(query_r_type)], dim=1))

        return self.final_score(final_query_embedding)


class KGCompletionGNN(nn.Module):
    def __init__(self, relation_feat, num_relations: int, input_dim: int, embed_dim: int, num_layers: int,
                 norm: int = 2, decoder: str = "MLP+TransE", dropout=0.5):
        super(KGCompletionGNN, self).__init__()

        local_vals = locals()
        self.instantiation_args = [local_vals[arg] for arg in inspect.signature(KGCompletionGNN).parameters]
        self.arg_signature = list(inspect.signature(KGCompletionGNN).parameters)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = nn.Dropout(p=dropout)
        self._encode_only = False

        self.language_transformer = BertModel.from_pretrained('bert-base-cased')

        self.relation_embedding = nn.Embedding(num_relations, input_dim)
        if relation_feat is not None:
            self.relation_embedding.weight = nn.Parameter(torch.tensor(relation_feat, dtype=torch.float))
        self.relation_embedding.weight.requires_grad = False  # Comment to learn through message passing relation embedding table
        self.relative_direction_embedding = nn.Embedding(2, embed_dim)
        self.head_or_tail_edge_embedding = nn.Embedding(2, embed_dim)

        self.edge_input_transform = nn.Linear(input_dim, embed_dim)

        self.entity_input_transform = nn.Linear(input_dim, embed_dim)

        self.norm_entity = nn.LayerNorm(embed_dim)
        self.norm_edge = nn.LayerNorm(embed_dim)

        self.message_passing_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassingLayer(embed_dim))
            self.edge_update_layers.append(EdgeUpdateLayer(embed_dim))

        self.relation_correlation_model = RelationCorrelationModel(num_relations, embed_dim)

        self.decoder = decoder
        self.classify_triple = TripleClassificationLayer(embed_dim)
        self.transE_decoder = TransEDecoder(num_relations, embed_dim)
        # self.conv_decoder = ConvolutionDecoder(embed_dim)

        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

    def encode_only(self, val: bool):
        self._encode_only = val
        return self

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, ht: Tensor,
                r_tensor: Tensor, r_query: Tensor, entity_feat: Tensor, query_nodes: Tensor, r_relative: Tensor,
                is_head_prediction: Tensor, queries=None, negative_targets=None, positive_targets=None, target_embeddings=None):
        """

        :param input_ids: For BERT
        :param token_type_ids: For BERT
        :param attention_mask: For BERT
        :param ht: edges in the message passing subgraphs
        :param r_tensor: relation types in the message passing subgraphs
        :param r_query: relation type of the query
        :param entity_feat: features of the entities in the message passing subgraphs
        :param query_nodes: the centroid nodes of each subgraph, entity associated with either being a target or a
                            query, not a sampled neighbor
        :param r_relative: direction of the edge in the subgraph relative to the query_node
        :param is_head_prediction: for each query, indicates whether it is a head prediction task
        :param queries: boolean indicating which query_nodes are part of a query (vs target)
        :param negative_targets: boolean indicating which query_nodes are negative targets
        :param positive_targets: boolean indicating which query_nodes are positive targets
        :param target_embeddings: for inference, pass a (N x d) tensor of target embeddings
        :return:
        """
        # Transform entities
        language_embedding = self.language_transformer(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[1]

        entity_feat[query_nodes] = language_embedding

        H_0 = self.act(self.entity_input_transform(self.dropout(entity_feat)))
        H_0 = self.norm_entity(H_0)
        H = H_0

        # Transform relations
        r_embed = self.relation_embedding(r_tensor)
        r_direction_embed = self.relative_direction_embedding(r_relative)
        E_0 = self.act(self.edge_input_transform(r_embed))
        E_0 = self.norm_edge(E_0)
        E = E_0 + r_direction_embed

        for i in range(self.num_layers):
            H = self.dropout(self.message_passing_layers[i](H, E, ht))
            E = self.edge_update_layers[i](H, E, ht)

        final_embeddings = H[query_nodes]  # TODO: look at pooling of message passing

        if self._encode_only:
            return final_embeddings

        if self.training:
            query_embeds = final_embeddings[queries]
            positive_target_embeds = final_embeddings[positive_targets]
            negative_target_embeds = final_embeddings[negative_targets]
        else:
            query_embeds = final_embeddings[queries]
            positive_target_embeds = final_embeddings[torch.logical_not(queries)]
            negative_target_embeds = target_embeddings

        if self.decoder == "TransE":
            pos_scores, neg_scores = self.transE_decoder(query_embeds, positive_target_embeds, negative_target_embeds, r_query, is_head_prediction)
        else:
            raise Exception(f"Decoder '{self.decoder}' not valid.")

        return pos_scores, neg_scores

    def get_loss_fn(self, margin=1.0):
        if self.decoder == "MLP":
            return nn.BCEWithLogitsLoss()
        elif self.decoder == "TransE":
            self.margin = margin
            return self.margin_ranking_loss
        elif self.decoder == "MLP+TransE":
            self.margin = margin
            return self.combo_loss
        elif self.decoder == "MLP+Conv":
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

    def margin_ranking_loss(self, pos_scores, neg_scores):
        """
        :param pos_scores: (num_pos,)
        :param neg_scores: (num_pos, num_neg)
        :return:
        """
        pos_scores = pos_scores.reshape(-1, 1).expand(-1, neg_scores.shape[1])

        target = torch.tensor([1], dtype=torch.long, device=pos_scores.device)
        return F.margin_ranking_loss(pos_scores, neg_scores, target, margin=self.margin)

    def combo_loss(self, pos_scores, neg_scores):
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

    def forward(self, query_embeds, pos_target_embeds, neg_target_embeds, r_type, is_head_prediction):
        """

        :param query_embeds: (Q,d)
        :param pos_target_embeds: (Q,d)
        :param neg_target_embeds: (num_negs,d)
        :param r_type: (Q,)
        :param is_head_prediction: (Q,) whether the query is associated with a head prediction task (hr->?) vs (tr->?)
        :return:
        """
        relation_embeds = self.relation_vector(r_type)

        head_embeds = torch.where(is_head_prediction.reshape(-1,1), pos_target_embeds, query_embeds)
        tail_embeds = torch.where(is_head_prediction.reshape(-1,1), query_embeds, pos_target_embeds)

        pos_distances = self.distance(head_embeds, relation_embeds, tail_embeds, self.distance_norm)

        neg_target_embeds = neg_target_embeds.reshape(1, -1, neg_target_embeds.shape[1])  # 1 x num negs x d
        query_embeds = query_embeds.reshape(-1, 1, query_embeds.shape[1])  # num queries x 1 x d
        relation_embeds = relation_embeds.reshape(-1, 1, relation_embeds.shape[1])  # num queries x 1 x d

        bkw_query_embeds = query_embeds[is_head_prediction]
        bkw_target_embeds = neg_target_embeds
        bkw_relation_embeds = relation_embeds[is_head_prediction]

        head_pred_distances = self.distance(bkw_target_embeds, bkw_relation_embeds, bkw_query_embeds, self.distance_norm)

        fwd_query_embeds = query_embeds[torch.logical_not(is_head_prediction)]
        fwd_target_embeds = neg_target_embeds
        fwd_relation_embeds = relation_embeds[torch.logical_not(is_head_prediction)]

        tail_pred_distances = self.distance(fwd_query_embeds, fwd_relation_embeds, fwd_target_embeds, self.distance_norm)

        neg_distances = torch.empty(query_embeds.shape[0], neg_target_embeds.shape[1], device=query_embeds.device)
        neg_distances[is_head_prediction] = head_pred_distances
        neg_distances[torch.logical_not(is_head_prediction)] = tail_pred_distances

        if torch.any(torch.isnan(pos_distances)):
            breakpoint()

        return -1*pos_distances, -1*neg_distances

    @staticmethod
    def distance(head_embeds, relation_embeds, tail_embeds, p):
        head_embeds = F.normalize(head_embeds, p=2, dim=-1)
        tail_embeds = F.normalize(tail_embeds, p=2, dim=-1)
        distances = torch.norm(head_embeds + relation_embeds - tail_embeds, p=p, dim=-1)
        return distances


class ConvolutionDecoder(nn.Module):
    def __init__(self, embed_dim, channels_1=32, channels_2=16, kernel_size=3, height_dim=32, dropout=0.2):
        super(ConvolutionDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.height_dim = height_dim
        self.channels_1 = channels_1
        self.channels_2 = channels_2
        self.kernel_size = kernel_size

        assert embed_dim % height_dim == 0, '{} does not divide {} without remainder'.format(height_dim, embed_dim)

        self.width_dim = int(self.embed_dim / self.height_dim)
        self.out_height = self.height_dim + 3 * (1 - self.kernel_size)
        self.out_width = self.width_dim + 3 * (1 - self.kernel_size)

        self.conv1 = nn.Conv2d(3, self.channels_1, kernel_size=(self.kernel_size, self.kernel_size))
        self.conv2 = nn.Conv2d(self.channels_1, self.channels_2, kernel_size=(self.kernel_size, self.kernel_size))
        self.conv3 = nn.Conv2d(self.channels_2, 1, kernel_size=(self.kernel_size, self.kernel_size))
        self.output = nn.Linear(self.out_height * self.out_width, 1)
        self.drop2d = nn.Dropout2d(p=dropout)
        self.drop = nn.Dropout(p=dropout)

        self.act = nn.ReLU()

    def forward(self, H, E, ht, queries):
        query_idxs = queries.nonzero().flatten()

        ht_q = ht[query_idxs]
        head_embs = H[ht_q[:, 0]].reshape(-1, self.height_dim, self.width_dim)
        tail_embs = H[ht_q[:, 1]].reshape(-1, self.height_dim, self.width_dim)
        relation_embs = E[query_idxs].reshape(-1, self.height_dim, self.width_dim)

        hrt_signal = torch.stack([head_embs, relation_embs, tail_embs], dim=1)
        hrt_signal = self.drop2d(hrt_signal)
        x = self.conv1(hrt_signal)
        x = self.act(x)
        x = self.drop2d(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop2d(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.reshape(x.shape[0], -1)
        x = self.output(x)

        return x
