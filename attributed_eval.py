import torch
import pandas as pd
import numpy as np
from ogb.lsc import WikiKG90MDataset


class AttributedEvaluator:

    def __init__(self):
        pass

    def eval(self, input_dict, stats):
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
        assert ('t_pred_top10' in input_dict['h,r->t']) \
               and ('hr' in input_dict['h,r->t']) \
               and ('t_candidate' in input_dict['h,r->t'])

        hr = input_dict['h,r->t']['hr']
        t_candidate = input_dict['h,r->t']['t_candidate']
        t_pred_top10 = input_dict['h,r->t']['t_pred_top10']
        t_correct_index = input_dict['h,r->t']['t_correct_index']

        if not isinstance(hr, torch.Tensor):
            hr = torch.from_numpy(hr).long()
        if not isinstance(t_candidate, torch.Tensor):
            t_candidate = torch.from_numpy(t_candidate).long()
        if not isinstance(t_pred_top10, torch.Tensor):
            t_pred_top10 = torch.from_numpy(t_pred_top10).long()
        if not isinstance(t_correct_index, torch.Tensor):
            t_correct_index = torch.from_numpy(t_correct_index).long()

        # Get rank
        nonzero_idx = torch.eq(t_pred_top10, t_correct_index.reshape(-1,1)).nonzero()
        t_ranks = torch.tensor([float('inf')]).repeat(t_pred_top10.shape[0])
        t_ranks[nonzero_idx[:,0]] = nonzero_idx[:,1].float()
        t_correct = t_candidate[np.arange(len(t_candidate)), t_correct_index]



        results = {}
        for name, col_stats in stats.items():
            col_name, stat_name = name.split('_')
            df = self._aggregate_by_stats(hr, t_correct, t_ranks, col_stats, col_name)
            results[name] = self._create_output_dict(df)
        return results

    @staticmethod
    def _create_output_dict(df):
        return df[['stats', 'mrr', 'hit', 'count']].to_dict('record')


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



    def _aggregate_by_object(self, hr, t, ranks, group_by_column):
        _data = torch.cat((hr, t[:, np.newaxis], ranks[:, np.newaxis]), dim=1).numpy()
        df = pd.DataFrame(data=_data, columns=['h', 'r', 't', 'ranks'])
        return df.groupby(group_by_column).agg(
                                    mrr=pd.NamedAgg(column='ranks', aggfunc=self._calculate_mrr),
                                    hit=pd.NamedAgg(column='ranks', aggfunc=self._calculate_hit),
                                    count=pd.NamedAgg(column='ranks', aggfunc='count')
                                        ).rename_axis('symbol').reset_index()


    def _aggregate_by_stats(self, hr, t, ranks, stats, group_by_column):
        _data = torch.cat((hr, t[:, np.newaxis], ranks[:, np.newaxis]), dim=1).numpy()
        df = pd.DataFrame(data=_data, columns=['h', 'r', 't', 'ranks'])
        df['h'] = df['h'].astype('int')
        df['r'] = df['r'].astype('int')
        df['t'] = df['t'].astype('int')
        df['stats'] = stats[df[group_by_column].values]
        return df.groupby('stats').agg(
            mrr=pd.NamedAgg(column='ranks', aggfunc=self._calculate_mrr),
            hit=pd.NamedAgg(column='ranks', aggfunc=self._calculate_hit),
            count=pd.NamedAgg(column='ranks', aggfunc='count')
        ).reset_index()

    def _calculate_mrr(self, ranks):
        rr = 1. / (ranks + 1.)
        return float(rr.mean().item())

    def _calculate_hit(self, ranks, k=10):
        hitk = np.sum(ranks.values + 1 <= k) / len(ranks)
        return hitk.item()

    def save_ranks(self, ranks):
        pass


def convert_stats_to_percentiles(stats: np.ndarray, thresholds=None):
    percentile_cutoffs = np.arange(10, 100, 10)
    if thresholds is None:
        thresholds = np.percentile(stats, percentile_cutoffs)
    binned_stats = np.digitize(stats, thresholds)
    return binned_stats, thresholds


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

    evaluator = AttributedEvaluator()

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


    rel_freq = np.random.randint(0, 200, dataset.num_relations)
    stats = {
        'r_freq': rel_freq
    }
    result = evaluator.eval(input_dict, stats)
    # print(result)

    # analyze results example
    # bins = list(range(1, 201))
    # rel_freq = np.random.randint(0, 200, dataset.num_relations)
    # evaluator.analyze_groups(result[2], stats=rel_freq, bins=bins)

    import IPython;
    IPython.embed()










