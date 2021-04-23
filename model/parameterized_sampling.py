import torch
import torch.nn as nn
#from utils.profile import profile

MAX_EDGES_TO_CONSIDER = 10000

class ParameterizedSampler(nn.Module):
    def __init__(self, num_query_relation_types, num_sample_relation_types):
        super(ParameterizedSampler, self).__init__()
        self.score_table = nn.Parameter(
            torch.ones((num_query_relation_types, num_sample_relation_types), dtype=torch.float))

    #@profile
    def forward(self, r_query, r_samples, num_samples, use_topk=False, replacement=True):
        """
        :param r_query: (int like)
        :param r_samples: np array indicating the relation type of edges (doubled to account for direction)
        :param num_samples: number of samples to take
        :param use_topk: whether to sample top k scores or sample based on distribution
        :param replacement: whether to sample with replacement when sampling using multinomial dist
        :return: selection array indicating sampled edges
        """
        assert not use_topk or not replacement, "If topk is being used, replacement must be False."

        if num_samples > r_samples.shape[0] and use_topk:
            num_samples = r_samples.shape[0]

        # if too many edges, do a uniform sample before parameterized sampling
        if r_samples.shape[0] > MAX_EDGES_TO_CONSIDER:
            indices = torch.randint(0, r_samples.shape[0], (MAX_EDGES_TO_CONSIDER,))
            r_samples = r_samples[indices]

        if r_samples.shape[0] == 0:
            use_topk = True
            num_samples = 0

        scores = self.score_table[r_query, r_samples]
        p = torch.softmax(scores, dim=0)
        if use_topk:
            p_x, x = torch.topk(p, k=num_samples)
        else:
            x = torch.multinomial(p, num_samples=num_samples, replacement=replacement)
            p_x = p[x]
        p_x = p_x / p_x.sum()
        return x, p_x
