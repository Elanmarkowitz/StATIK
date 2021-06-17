import tarfile

import pandas as pd
import numpy as np
import os
import pickle

from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer

import urllib.request


ROOT_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/mehrnoom"

DATASET_INFO = dt_info = {
        'dataset': 'WN18RR',
        'url': 'https://surfdrive.surf.nl/files/index.php/s/N1c8VRH0I6jTJuN/download',
        'train': 'ind-train.tsv',
        'test': 'ind-test.tsv',
        'dev': 'ind-dev.tsv',
        'ent_desc': 'entity2text.txt',
        'rel_desc': 'relation2text.txt'
    }

class ProcessWordNet(object):
    def __init__(self, root_data_dir=None, dataset_info=None):
        # print('hello')
        # try:
        self.num_relations = None
        self.num_entities = None
        self.relation_feat = None
        self.entity_feat = None
        self.train_hrt = None
        self.valid_hrt = None
        self.test_hrt = None
        self.entity_descs = None
        self.relation_descs = None
        self.entity2id = None
        self.relation2id = None

        self.dataset_info = DATASET_INFO if dataset_info is None else dataset_info
        if root_data_dir is None:
            self.data_dir = os.path.join(ROOT_DIR, self.dataset_info['dataset'])
        else:
            self.data_dir = os.path.join(root_data_dir, 'WN18RR')

        if not os.path.isdir(self.data_dir):
            print('Downloading data ....')
            self.download_data(self.dataset_info['url'], self.dataset_info['dataset'])

        if os.path.isdir(os.path.join(self.data_dir, 'processed')):
            print('Loading data ....')
            self.load_data(self.data_dir)
        else:
            print('Processing data ...')
            self.load_process_data()


    @staticmethod
    def download_data(url, dataset):
        with urllib.request.urlopen(url) as dl_file:
            with open(os.path.join(ROOT_DIR,  dataset), 'wb') as out_file:
                out_file.write(dl_file.read())
        tar = tarfile.open(os.path.join(ROOT_DIR, dataset), "r:gz")
        tar.extractall(ROOT_DIR)
        tar.close()



    def read_triples(self, filename):
        triples = pd.read_csv(os.path.join(self.data_dir, filename), names=['h', 'r', 't'], sep='\t')
        return triples


    def read_descriptions(self):
        ent_desc = pd.read_csv(os.path.join(self.data_dir, self.dataset_info['ent_desc']), names=['code', 'description'], sep='\t')
        rel_desc = pd.read_csv(os.path.join(self.data_dir, self.dataset_info['rel_desc']), names=['code', 'description'], sep='\t')
        return ent_desc, rel_desc


    def write_to_npy(self, np_array, filename):
        np.save(os.path.join(self.data_dir, 'processed', filename[:-len('txt')] + 'npy'), np_array)


    def load_from_npy(self, filename):
        return np.load(os.path.join(self.data_dir, 'processed', filename[:-len('txt')] + 'npy'), allow_pickle=True)

    def load_process_data(self):
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        if not os.path.isdir(os.path.join(self.data_dir, 'processed')):
            os.mkdir(os.path.join(self.data_dir, 'processed'))

        self.entity_descs, self.relation_descs = self.read_descriptions()
        self.get_entity_features()
        self.write_to_npy(self.entity_feat, 'entity_features.npy')
        self.write_to_npy(self.relation_feat, 'relation_features.npy')

        # read data
        self.train_hrt = self.read_triples(self.dataset_info['train'])
        self.valid_hrt = self.read_triples(self.dataset_info['dev'])
        self.test_hrt = self.read_triples(self.dataset_info['test'])

        # create symbols to id
        self.create_symbols_to_id()
        self.num_entities = len(self.entity2id.keys())
        self.num_relations = len(self.relation2id.keys())

        to_replace_dct = {
            'h': self.entity2id,
            'r': self.relation2id,
            't': self.entity2id}

        self.train_hrt = np.asarray(self.train_hrt.replace(to_replace=to_replace_dct).values)
        self.valid_hrt = np.asarray(self.valid_hrt.replace(to_replace=to_replace_dct).values)
        self.test_hrt = np.asarray(self.test_hrt.replace(to_replace=to_replace_dct).values)

        with open(os.path.join(self.data_dir, 'processed', 'ent2id.pkl'), 'wb') as fp:
            pickle.dump(self.entity2id, fp)

        with open(os.path.join(self.data_dir, 'processed', 'rel2id.pkl'), 'wb') as fp:
            pickle.dump(self.relation2id, fp)

        self.write_to_npy(self.train_hrt, self.dataset_info['train'])
        self.write_to_npy(self.valid_hrt, self.dataset_info['dev'])
        self.write_to_npy(self.test_hrt, self.dataset_info['test'])

    def load_data(self, data_dir: str):
        with open(os.path.join(data_dir, 'processed', 'ent2id.pkl'), 'rb') as fp:
            self.entity2id = pickle.load(fp)

        with open(os.path.join(data_dir, 'processed', 'rel2id.pkl'), 'rb') as fp:
            self.relation2id = pickle.load(fp)

        self.num_entities = len(self.entity2id.keys())
        self.num_relations = len(self.relation2id.keys())

        self.train_hrt = self.load_from_npy(self.dataset_info['train'])
        self.valid_hrt = self.load_from_npy(self.dataset_info['dev'])
        self.test_hrt = self.load_from_npy(self.dataset_info['test'])

        self.entity_feat = self.load_from_npy('entity_features.npy')
        self.relation_feat = self.load_from_npy('relation_features.npy')

    def create_symbols_to_id(self):
        # self.entity2id = defaultdict(int)
        # for idx, wn_id in enumerate(self.entity_descs['code'].values):
        #     self.entity2id[wn_id] = idx
        self.entity2id = defaultdict(int)
        train_ent = set(self.train_hrt['h'].values)
        train_ent.update(set(self.train_hrt['t'].values))
        for idx, wn_id in enumerate(train_ent):
            self.entity2id[wn_id] = idx

        dev_ent = set(self.valid_hrt['h'].values)
        dev_ent.update(self.valid_hrt['t'].values)
        for idx, wn_id in enumerate(dev_ent):
            self.entity2id[wn_id] = idx

        test_ent = set(self.test_hrt['h'].values)
        test_ent.update(self.test_hrt['t'].values)
        for idx, wn_id in enumerate(test_ent):
            self.entity2id[wn_id] = idx



        self.relation2id = defaultdict(int)
        for idx, wn_id in enumerate(self.relation_descs['code'].values):
            self.relation2id[wn_id] = idx

        # import IPython;IPython.embed()s

    def get_entity_features(self):
        print('Creating features using language model.')
        model = SentenceTransformer('stsb-distilroberta-base-v2')
        self.entity_feat = model.encode(self.entity_descs['description'].values)
        self.relation_feat = model.encode(self.relation_descs['description'].values)


if __name__ == "__main__":
    dt_info = {
        'dataset': 'WN18RR',
        'url': 'https://surfdrive.surf.nl/files/index.php/s/N1c8VRH0I6jTJuN/download',
        'train': 'ind-train.tsv',
        'test': 'ind-test.tsv',
        'dev': 'ind-dev.tsv',
        'ent_desc': 'entity2text.txt',
        'rel_desc': 'relation2text.txt'

    }
    wdn = ProcessWordNet(dt_info)



