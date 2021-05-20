import torch
import pandas as pd
import numpy as np
from ogb.lsc import WikiKG90MDataset


class AttibutedEvaluator:

    def __init__(self):
        pass

    def eval(self, input_dict):
        '''
            Format of input_dict:
            - 'h,r->t'
                - t_pred: np.ndarray of shape (num_eval_triplets, n_entities)
                    (i,j) represents the j-th prediction for i-th triplet
                - t_correct_index: np.ndarray of shape (num_eval_triplets,)
            - 'hr'
                np.ndarray of shape (num_eval_triplets, 2)

        '''
        assert 'h,r->t' in input_dict
        assert ('t_pred' in input_dict['h,r->t']) \
               and ('t_correct_index' in input_dict['h,r->t']) \
               and ('hr' in input_dict['h,r->t']) \
               and ('t_candidate' in input_dict['h,r->t'])

        hr = input_dict['h,r->t']['hr']
        t_candidate = input_dict['h,r->t']['t_candidate']
        t_pred = input_dict['h,r->t']['t_pred']
        t_correct_index = input_dict['h,r->t']['t_correct_index']

        if not isinstance(hr, torch.Tensor):
            hr = torch.from_numpy(hr)
        if not isinstance(t_candidate, torch.Tensor):
            t_candidate = torch.from_numpy(t_candidate)
        if not isinstance(t_pred, torch.Tensor):
            t_pred = torch.from_numpy(t_pred)
        if not isinstance(t_correct_index, torch.Tensor):
            t_correct_index = torch.from_numpy(t_correct_index)

        # Get rank
        t_ranks = torch.argsort(t_pred, dim=1)[np.arange(len(t_pred)), t_correct_index]
        t_correct = t_candidate[np.arange(len(t_candidate)), t_correct_index]
        return self._group_by(hr, t_correct, t_ranks, 'h'), \
               self._group_by(hr, t_correct, t_ranks, 't'), \
               self._group_by(hr, t_correct, t_ranks, 'r')

        # g = self._group_by(hr, t_correct, t_ranks, 'r')
        # bins = [0, 30, 50, 100, 200]
        # stats = np.random.randint(0, 200, dataset.num_relations)
        # self.analyze_groups(g, stats, bins)

    def analyze_groups(self, groups, stats, bins):
        '''
        :param groups: pandas data frame, index='r', columns=['scores']
            - groups['score'] is a tuple (score, group_size)
        :param values: np.ndarray of size (1, num_relations) and contains either relation frequencies or entity degrees
        :param bins: If we want to group relation frequencies by beans
        :return:
        '''

        groups['stats'] = stats[groups['symbol'].values]
        group_obj = groups.groupby(pd.cut(groups.stats, bins))
        agg_groups = group_obj.apply(self._aggregate_groups)

        return group_obj, agg_groups

    def _aggregate_groups(self, df):
        mrr = np.dot(df['mrr'].values, df['count'].values) / df['count'].values.sum()
        hit = np.dot(df['hit'].values, df['count'].values) / df['count'].values.sum()
        return pd.Series([mrr, hit], index=["mrr", "hit"])


    def _group_by(self, hr, t, ranks, group_by_column):
        _data = torch.cat((hr, t[:, np.newaxis], ranks[:, np.newaxis]), dim=1).numpy()
        df = pd.DataFrame(data=_data, columns=['h', 'r', 't', 'ranks'])
        return df.groupby(group_by_column).agg(
                                    mrr=pd.NamedAgg(column='ranks', aggfunc=self._calculate_mrr),
                                    hit=pd.NamedAgg(column='ranks', aggfunc=self._calculate_hit),
                                    count=pd.NamedAgg(column='ranks', aggfunc='count')
                                        ).rename_axis('symbol').reset_index()



    def _calculate_mrr(self, ranks):
        rr = 1. / (ranks + 1.)
        return float(rr.mean().item())

    def _calculate_hit(self, ranks, k=10):
        hitk = np.sum(ranks.values + 1 <= k) / len(ranks)
        return hitk.item()

    def save_ranks(self, ranks):
        pass


if __name__ == '__main__':
    dataset = WikiKG90MDataset(root='/data/elanmark/')
    # print(dataset)
    # print(dataset.num_entities)
    # print(dataset.entity_feat)
    # print(dataset.entity_feat.shape)
    # print(dataset.num_relations)
    # print(dataset.relation_feat)
    # print(dataset.all_relation_feat)
    # print(dataset.relation_feat.shape)
    # print(dataset.train_hrt)
    # print(dataset.valid_dict)
    # print(dataset.test_dict)
    # print(dataset.valid_dict['h,r->t']['t_correct_index'].max())
    # print(dataset.valid_dict['h,r->t']['t_correct_index'].min())

    evaluator = AttibutedEvaluator()

    valid_dict = dataset.valid_dict
    t_correct_index = valid_dict['h,r->t']['t_correct_index']
    test_task = dataset.test_dict['h,r->t']

    # t_correct_index = test_task['t_correct_index'] # key error
    hr = valid_dict['h,r->t']['hr']
    # t_candidate = test_task['t_candidate']
    t_pred_top10 = np.random.rand(len(t_correct_index), 1001)

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': t_correct_index, 't_pred': t_pred_top10}
    input_dict['h,r->t']['hr'] = hr
    input_dict['h,r->t']['t_candidate'] = valid_dict['h,r->t']['t_candidate']
    result = evaluator.eval(input_dict)
    print(result)

    t_correct_index = torch.from_numpy(t_correct_index)
    t_pred_top10 = torch.from_numpy(t_pred_top10)

    input_dict = {}
    input_dict['h,r->t'] = {'t_correct_index': t_correct_index, 't_pred_top10': t_pred_top10}
    result = evaluator.eval(input_dict)
    print(result)

    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': np.random.randint(0, 1001, size=(1359303, 10))}
    # evaluator.save_test_submission(input_dict, 'result')

    # print(dataset.all_entity_feat)
    # print(dataset.all_entity_feat.shape)










