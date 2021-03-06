import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from transformers import AutoModel
import torch_scatter

ENCODERS = ['ours_parallel', 'ours_sequential', 'BLP', 'StAR']


class ModelConfig:
    def __init__(self, FLAGS):
        self.encoder = FLAGS.encoder
        self.decoder = FLAGS.decoder
        self.layers = FLAGS.layers
        self.embed_dim = FLAGS.embed_dim
        self.language_model = FLAGS.language_model
        self.dropout = FLAGS.dropout
        self.train_through_relation_feat = False


class MessageCalculationLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(MessageCalculationLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transform_message = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, H: Tensor, E: Tensor, heads: Tensor):
        """
        :param H: shape n x embed_dim
        :param E: shape m x embed_dim
        :param heads: shape m x 1
        :return:
            processed messages for nodes. shape nodes_in_batch x embed_dim
        """
        H_heads = H[heads]
        raw_messages = torch.cat([H_heads, E, H_heads + E, H_heads * E], dim=1)
        messages = self.transform_message(raw_messages)

        # TODO: Maybe normalize
        return messages


class MessagePassingLayer(nn.Module):
    def __init__(self, embed_dim: int, agg_method="mean"):
        assert agg_method in ["sum", "mean"]
        super(MessagePassingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.calc_messages_fwd = MessageCalculationLayer(embed_dim)
        self.calc_messages_back = MessageCalculationLayer(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.LeakyReLU()
        self.agg_method = agg_method

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
        agg_messages = torch.zeros((nodes_in_batch, messages_fwd.shape[1]), dtype=messages_fwd.dtype, device=device)
        if self.agg_method == "sum":
            torch_scatter.scatter_add(messages, index=msg_dst, dim=0, out=agg_messages)
        elif self.agg_method == "mean":
            torch_scatter.scatter_mean(messages, index=msg_dst, dim=0, out=agg_messages)
        else:
            raise AssertionError("agg method not known.")

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
    def __init__(self, relation_feat, num_relations: int, input_dim: int, config: ModelConfig):
        super(KGCompletionGNN, self).__init__()

        local_vals = locals()
        self.instantiation_args = [local_vals[arg] for arg in inspect.signature(KGCompletionGNN).parameters]
        self.arg_signature = list(inspect.signature(KGCompletionGNN).parameters)

        self.language_model = config.language_model
        self.encoder = config.encoder
        self.decoder = config.decoder
        assert self.encoder in ENCODERS, f'encoder must be one of {ENCODERS}'

        self.embed_dim = config.embed_dim
        self.num_layers = config.layers
        self.dropout = nn.Dropout(p=config.dropout)
        self._encode_only = False

        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

        self.language_transformer = AutoModel.from_pretrained(self.language_model)

        self.relation_embedding = nn.Embedding(num_relations, input_dim)
        if relation_feat is not None:
            self.relation_embedding.weight = nn.Parameter(torch.tensor(relation_feat, dtype=torch.float))
        if not config.train_through_relation_feat:
            self.relation_embedding.weight.requires_grad = False   # Comment to learn through message passing relation embedding table
        self.relative_direction_embedding = nn.Embedding(2, self.embed_dim)
        self.head_or_tail_edge_embedding = nn.Embedding(2, self.embed_dim)

        self.edge_input_transform = nn.Linear(input_dim, self.embed_dim)

        self.entity_input_transform = nn.Linear(input_dim, self.embed_dim)
        self.entity_input_transform2 = nn.Linear(input_dim, self.embed_dim)

        self.norm_entity = nn.LayerNorm(self.embed_dim)
        self.norm_edge = nn.LayerNorm(self.embed_dim)

        self.message_passing_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.message_passing_layers.append(MessagePassingLayer(self.embed_dim))
            self.edge_update_layers.append(EdgeUpdateLayer(self.embed_dim))

        self.mlp_decoder = MLPClassificationLayer(self.embed_dim)
        self.transE_decoder = TransEDecoder(num_relations, self.embed_dim)
        self.complex_decoder = ComplExDecoder(num_relations, self.embed_dim)
        self.distmult_decoder = DistMultDecoder(num_relations, self.embed_dim)
        self.language_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mp_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_combination_layer = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim),
                                                       self.act,
                                                       nn.Linear(self.embed_dim, self.embed_dim))
        # self.conv_decoder = ConvolutionDecoder(embed_dim)

    def encode_only(self, val: bool):
        self._encode_only = val
        return self

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, ht: Tensor,
                r_tensor: Tensor, r_query: Tensor, entity_feat: Tensor, query_nodes: Tensor, r_relative: Tensor,
                is_head_prediction: Tensor, queries=None, negative_targets=None, positive_targets=None,
                target_embeddings=None):
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
        language_embedding = self.language_transformer(input_ids=input_ids, token_type_ids=token_type_ids,
                                                       attention_mask=attention_mask)[0][:, 0]

        if self.encoder == "ours_sequential":
            entity_feat[query_nodes] = language_embedding

        if self.encoder in ['ours_parallel', 'ours_sequential']:

            H_0 = self.entity_input_transform2(self.dropout(entity_feat))
            H_0 = self.norm_entity(H_0)
            H = H_0
            #
            # Transform relations
            r_embed = self.relation_embedding(r_tensor)
            r_direction_embed = self.relative_direction_embedding(r_relative)
            E_0 = self.act(self.edge_input_transform(r_embed))
            E_0 = self.norm_edge(E_0)
            E = E_0 + r_direction_embed

            for i in range(self.num_layers):
                H = self.dropout(self.message_passing_layers[i](H, E, ht))
                E = self.edge_update_layers[i](H, E, ht)

        # final_embeddings = 0.5*H[query_nodes] + 0.5*H_0[query_nodes]  # TODO: look at pooling of message passing

        if self.encoder == "ours_sequential":
            final_embeddings = H[query_nodes]
        elif self.encoder == "ours_parallel":
            catted_embeds = torch.cat([H[query_nodes],
                                      self.entity_input_transform(language_embedding)], dim=-1)
            final_embeddings = self.encoder_combination_layer(self.act(catted_embeds))
            # final_embeddings = self.mp_layer_norm(H[query_nodes]) + self.language_layer_norm(self.entity_input_transform(language_embedding))
            # final_embeddings = F.normalize(H[query_nodes]) + F.normalize(self.entity_input_transform(language_embedding))
            # final_embeddings = H[query_nodes] + self.entity_input_transform(language_embedding)
        elif self.encoder == "BLP":
            final_embeddings = self.entity_input_transform(language_embedding)
        elif self.encoder == 'StAR':
            final_embeddings = self.entity_input_transform(language_embedding)  # TODO: check that this is how they handle
        else:
            raise Exception(f'Unknown handling of {self.encoder}, check code for issue')

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
            pos_scores, neg_scores = self.transE_decoder(query_embeds, positive_target_embeds, negative_target_embeds,
                                                         r_query, is_head_prediction, add_batch_to_negs=self.training)
            return pos_scores, neg_scores
        elif self.decoder == "ComplEx":
            pos_scores, neg_scores = self.complex_decoder(query_embeds, positive_target_embeds, negative_target_embeds,
                                                         r_query, is_head_prediction, add_batch_to_negs=self.training)
            return pos_scores, neg_scores
        elif self.decoder == "DistMult":
            pos_scores, neg_scores = self.distmult_decoder(query_embeds, positive_target_embeds, negative_target_embeds,
                                                          r_query, is_head_prediction, add_batch_to_negs=self.training)
            return pos_scores, neg_scores

        elif self.decoder == "MLP":
            pos_scores, neg_scores = self.mlp_decoder(query_embeds, positive_target_embeds, negative_target_embeds,
                                                      is_head_prediction, add_batch_to_negs=self.training)
            return pos_scores, neg_scores
        elif self.decoder in ["MLP+TransE", "TransE+MLP"]:
            transe_pos_scores, transe_neg_scores = self.transE_decoder(query_embeds, positive_target_embeds,
                                                                       negative_target_embeds, r_query,
                                                                       is_head_prediction, add_batch_to_negs=self.training)
            mlp_pos_scores, mlp_neg_scores = self.mlp_decoder(query_embeds, positive_target_embeds,
                                                              negative_target_embeds, is_head_prediction,
                                                              add_batch_to_negs=self.training)
            if self.training:
                return (transe_pos_scores, transe_neg_scores), (mlp_pos_scores, mlp_neg_scores)
            else:
                return (mlp_pos_scores, mlp_neg_scores) if self.decoder == "MLP+TransE" \
                    else (transe_pos_scores, transe_neg_scores)
        else:
            raise Exception(f"Decoder '{self.decoder}' not valid.")

    def get_loss_fn(self, margin=1.0):
        if self.decoder == "MLP":
            return self.bce_with_logits
        elif self.decoder == "TransE":
            self.margin = margin
            return self.margin_ranking_loss
        elif self.decoder == "DistMult":
            self.margin = margin
            return self.margin_ranking_loss
        elif self.decoder == "ComplEx":
            self.margin = margin
            return self.margin_ranking_loss
        elif self.decoder in ["MLP+TransE", "TransE+MLP"]:
            self.margin = margin
            return self.combo_loss
        else:
            Exception(f"Loss function not known for {self.decoder}")

    def margin_ranking_loss(self, pos_scores, neg_scores, neg_filter=None):
        """
        :param pos_scores: (num_pos,)
        :param neg_scores: (num_pos, num_neg)
        :return:
        """
        pos_scores = pos_scores.reshape(-1, 1).expand(-1, neg_scores.shape[1])

        target = torch.tensor([1], dtype=torch.long, device=pos_scores.device)
        if neg_filter is not None:
            pos_scores = pos_scores.flatten()[neg_filter.flatten()]
            neg_scores = neg_scores.flatten()[neg_filter.flatten()]
        return F.margin_ranking_loss(pos_scores, neg_scores, target, margin=self.margin)

    @staticmethod
    def bce_with_logits(pos_scores, neg_scores, neg_filter=None):
        if neg_filter is not None:
            pos_scores = pos_scores.flatten()[neg_filter.flatten()]
            neg_scores = neg_scores.flatten()[neg_filter.flatten()]
        pos_scores = pos_scores.flatten()
        neg_scores = neg_scores.flatten()
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def combo_loss(self, transe_scores, mlp_scores, neg_filter=None):
        bce_loss = self.bce_with_logits(mlp_scores[0], mlp_scores[1], neg_filter=neg_filter)
        transe_loss = self.margin_ranking_loss(transe_scores[0], transe_scores[1], neg_filter=neg_filter)
        return transe_loss + bce_loss


def query_target_to_head_tail(query_embeds: Tensor, pos_target_embeds: Tensor, neg_target_embeds: Tensor,
                              is_head_prediction: Tensor, add_pos_to_negs=False):
    # num_negs = neg_target_embeds.shape[0]
    num_q = query_embeds.shape[0]
    embed_dim = query_embeds.shape[1]

    head_embeds = torch.where(is_head_prediction.reshape(-1, 1), pos_target_embeds, query_embeds)
    tail_embeds = torch.where(is_head_prediction.reshape(-1, 1), query_embeds, pos_target_embeds)

    neg_target_embeds = neg_target_embeds.reshape(1, -1, embed_dim).expand(num_q, -1, -1)  # num q x num negs x d

    if add_pos_to_negs:
        NUM_NEGS = 0
        neg_targets_from_pos = pos_target_embeds.reshape(1, num_q, embed_dim).expand(num_q, -1, -1)
        # neg_targets_from_pos = neg_targets_from_pos[torch.logical_not(torch.eye(num_q))].reshape(num_q, num_q-1, embed_dim)
        # random_select = torch.randint(0, num_q - 1, (num_q, NUM_NEGS))
        # neg_targets_from_pos = neg_targets_from_pos[torch.arange(num_q).reshape(-1, 1), random_select]
        neg_target_embeds = torch.cat([neg_target_embeds, neg_targets_from_pos], dim=1)  # num q x num neg + num q - 1 x d

    query_embeds = query_embeds.reshape(-1, 1, embed_dim).expand(-1, neg_target_embeds.shape[1], -1)  # num queries x num neg + num q - 1 x d

    neg_head_embeds = torch.where(is_head_prediction.reshape(-1, 1, 1), neg_target_embeds, query_embeds)
    neg_tail_embeds = torch.where(is_head_prediction.reshape(-1, 1, 1), query_embeds, neg_target_embeds)

    return head_embeds, tail_embeds, neg_head_embeds, neg_tail_embeds


class MLPClassificationLayer(nn.Module):
    def __init__(self, embed_dim):
        super(MLPClassificationLayer, self).__init__()
        self.proj = nn.Linear(4 * embed_dim, 1)

    def forward(self, query_embeds, pos_target_embeds, neg_target_embeds, is_head_prediction,
                add_batch_to_negs=False):
        head_embeds, tail_embeds, neg_head_embeds, neg_tail_embeds = query_target_to_head_tail(
            query_embeds, pos_target_embeds, neg_target_embeds, is_head_prediction,
            add_pos_to_negs=add_batch_to_negs)
        pos = torch.cat([head_embeds, tail_embeds, head_embeds - tail_embeds, head_embeds * tail_embeds], dim=-1)
        neg = torch.cat([neg_head_embeds, neg_tail_embeds, neg_head_embeds - neg_tail_embeds,
                         neg_head_embeds * neg_tail_embeds], dim=-1)
        pos_scores = self.proj(pos)
        neg_scores = self.proj(neg)
        return pos_scores, neg_scores


class TransEDecoder(nn.Module):
    def __init__(self, num_relations: int, embed_dim: int, distance_norm: int = 1):
        super(TransEDecoder, self).__init__()
        self.distance_norm = distance_norm
        self.relation_vector = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_vector.weight.data)

    def forward(self, query_embeds, pos_target_embeds, neg_target_embeds, r_type, is_head_prediction,
                add_batch_to_negs=False):
        """
        :param query_embeds: (Q,d)
        :param pos_target_embeds: (Q,d)
        :param neg_target_embeds: (num_negs,d)
        :param r_type: (Q,)
        :param is_head_prediction: (Q,) whether the query is associated with a head prediction task (hr->?) vs (tr->?)
        :param add_batch_to_negs: Whether to use the pos_targets in the batch as negative targets for the rest of the
                                  batch. Should b used for training only.
        :return:
        """
        relation_embeds = self.relation_vector(r_type)

        head_embeds, tail_embeds, neg_head_embeds, neg_tail_embeds = query_target_to_head_tail(
            query_embeds, pos_target_embeds, neg_target_embeds, is_head_prediction,
            add_pos_to_negs=add_batch_to_negs)

        pos_distances = self.distance(head_embeds, relation_embeds, tail_embeds, self.distance_norm)

        relation_embeds = relation_embeds.reshape(-1, 1, relation_embeds.shape[1])  # num queries x 1 x d

        neg_distances = self.distance(neg_head_embeds, relation_embeds, neg_tail_embeds, self.distance_norm)

        return -1*pos_distances, -1*neg_distances

    @staticmethod
    def distance(head_embeds, relation_embeds, tail_embeds, p):
        head_embeds = F.normalize(head_embeds, p=2, dim=-1)
        tail_embeds = F.normalize(tail_embeds, p=2, dim=-1)
        distances = torch.norm(head_embeds + relation_embeds - tail_embeds, p=p, dim=-1)
        return distances


class ComplExDecoder(nn.Module):
    def __init__(self, num_relations: int, embed_dim: int, distance_norm: int = 1):
        super(ComplExDecoder, self).__init__()
        self.distance_norm = distance_norm
        self.relation_vector = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_vector.weight.data)

    def forward(self, query_embeds, pos_target_embeds, neg_target_embeds, r_type, is_head_prediction,
                add_batch_to_negs=False):
        """
        :param query_embeds: (Q,d)
        :param pos_target_embeds: (Q,d)
        :param neg_target_embeds: (num_negs,d)
        :param r_type: (Q,)
        :param is_head_prediction: (Q,) whether the query is associated with a head prediction task (hr->?) vs (tr->?)
        :param add_batch_to_negs: Whether to use the pos_targets in the batch as negative targets for the rest of the
                                  batch. Should b used for training only.
        :return:
        """
        relation_embeds = self.relation_vector(r_type)

        head_embeds, tail_embeds, neg_head_embeds, neg_tail_embeds = query_target_to_head_tail(
            query_embeds, pos_target_embeds, neg_target_embeds, is_head_prediction,
            add_pos_to_negs=add_batch_to_negs)

        pos_distances = self.distance(head_embeds, relation_embeds, tail_embeds, self.distance_norm)

        relation_embeds = relation_embeds.reshape(-1, 1, relation_embeds.shape[1])  # num queries x 1 x d

        neg_distances = self.distance(neg_head_embeds, relation_embeds, neg_tail_embeds, self.distance_norm)

        return -1*pos_distances, -1*neg_distances

    @staticmethod
    def distance(head_embeds, relation_embeds, tail_embeds, p):
        head_embeds = F.normalize(head_embeds, p=2, dim=-1)
        tail_embeds = F.normalize(tail_embeds, p=2, dim=-1)

        h_Re, h_Im = head_embeds[..., :int(head_embeds.shape[-1] / 2)], head_embeds[..., int(head_embeds.shape[-1] / 2):]
        t_Re, t_Im = tail_embeds[..., :int(tail_embeds.shape[-1] / 2)], tail_embeds[..., int(tail_embeds.shape[-1] / 2):]
        r_Re, r_Im = relation_embeds[..., :int(relation_embeds.shape[-1] / 2)], relation_embeds[..., int(relation_embeds.shape[-1]/2):]

        distances = torch.sum(r_Re * h_Re * t_Re, dim=-1) + \
                    torch.sum(r_Re * h_Im * t_Im, dim=-1) + \
                    torch.sum(r_Im * h_Re * t_Im, dim=-1) - \
                    torch.sum(r_Im * h_Im * t_Im, dim=-1)
        return distances


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, embed_dim: int, distance_norm: int = 1):
        super(DistMultDecoder, self).__init__()
        self.distance_norm = distance_norm
        self.relation_vector = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_vector.weight.data)

    def forward(self, query_embeds, pos_target_embeds, neg_target_embeds, r_type, is_head_prediction,
                add_batch_to_negs=False):
        """
        :param query_embeds: (Q,d)
        :param pos_target_embeds: (Q,d)
        :param neg_target_embeds: (num_negs,d)
        :param r_type: (Q,)
        :param is_head_prediction: (Q,) whether the query is associated with a head prediction task (hr->?) vs (tr->?)
        :param add_batch_to_negs: Whether to use the pos_targets in the batch as negative targets for the rest of the
                                  batch. Should b used for training only.
        :return:
        """
        relation_embeds = self.relation_vector(r_type)

        head_embeds, tail_embeds, neg_head_embeds, neg_tail_embeds = query_target_to_head_tail(
            query_embeds, pos_target_embeds, neg_target_embeds, is_head_prediction,
            add_pos_to_negs=add_batch_to_negs)

        import IPython; IPython.embed()
        pos_distances = self.distance(head_embeds, relation_embeds, tail_embeds, self.distance_norm)

        relation_embeds = relation_embeds.reshape(-1, 1, relation_embeds.shape[1])  # num queries x 1 x d

        neg_distances = self.distance(neg_head_embeds, relation_embeds, neg_tail_embeds, self.distance_norm)

        return -1*pos_distances, -1*neg_distances

    @staticmethod
    def distance(head_embeds, relation_embeds, tail_embeds, p):
        head_embeds = F.normalize(head_embeds, p=2, dim=-1)
        tail_embeds = F.normalize(tail_embeds, p=2, dim=-1)

        distances = torch.sum(head_embeds * relation_embeds * tail_embeds, dim=-1)

        return distances
