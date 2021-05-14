import tarfile

import pandas as pd
import numpy as np
import os
import pickle

from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer

import urllib.request


ROOT_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/elanmark"
DATA_DIR = os.path.join(ROOT_DIR, 'wordnet-mlj12')


class ProcessWordNet(object):
    def __init__(self, root_data_dir: str = None):
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
        if root_data_dir is not None:
            self.load_data(os.path.join(root_data_dir, 'wordnet-mlj12'))
        elif os.path.isdir(os.path.join(DATA_DIR, 'processed')):
            self.load_data(DATA_DIR)
        else:
            self.load_process_data()
        # except:
        #     self.download_data()

    @staticmethod
    def download_data():
        print('Downloading dataset.')
        url = 'https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz'
        with urllib.request.urlopen(url) as dl_file:
            with open(os.path.join(ROOT_DIR, 'wordnet.tar.gz'), 'wb') as out_file:
                out_file.write(dl_file.read())
        tar = tarfile.open(os.path.join(ROOT_DIR, 'wordnet.tar.gz'), "r:gz")
        tar.extractall(ROOT_DIR)
        tar.close()


    @staticmethod
    def read_triples(filename, to_replace):
        triples = pd.read_csv(os.path.join(DATA_DIR, filename), names=['h', 'r', 't'], sep='\t').replace(to_replace)
        return triples

    @staticmethod
    def read_descriptions():
        if not os.path.isfile(os.path.join(DATA_DIR, 'wordnet-mlj12-definitions.txt')):
            ProcessWordNet.download_data()
        ent_desc = pd.read_csv(os.path.join(DATA_DIR, 'wordnet-mlj12-definitions.txt'), names=['code', 'name', 'description'], sep='\t')
        rel_desc = pd.read_csv(os.path.join(DATA_DIR, 'wordnet-mlj12-train.txt'), names=['h', 'description', 'r'], sep='\t')['description'].drop_duplicates()
        return ent_desc, rel_desc

    @staticmethod
    def write_to_npy(np_array, filename):
        np.save(os.path.join(DATA_DIR, 'processed', filename), np_array)

    @staticmethod
    def load_from_npy(filename):
        return np.load(os.path.join(DATA_DIR, 'processed', filename), allow_pickle=True)

    def load_process_data(self):
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.isdir(os.path.join(DATA_DIR, 'processed')):
            os.mkdir(os.path.join(DATA_DIR, 'processed'))

        self.entity_descs, self.relation_descs = self.read_descriptions()
        self.get_entity_features()
        self.write_to_npy(self.entity_feat, 'entity_features.npy')
        self.write_to_npy(self.relation_feat, 'relation_features.npy')

        # create symbols to id
        self.create_symbols_to_id()
        self.num_entities = len(self.entity2id.keys())
        self.num_relations = len(self.relation2id.keys())

        # read train data
        to_replace_dct = {
            'h': self.entity2id,
            'r': self.relation2id,
            't': self.entity2id}
        self.train_hrt = np.array(self.read_triples('wordnet-mlj12-train.txt', to_replace=to_replace_dct).values)
        self.valid_hrt = np.array(self.read_triples('wordnet-mlj12-valid.txt', to_replace=to_replace_dct).values)
        self.test_hrt = np.array(self.read_triples('wordnet-mlj12-test.txt', to_replace=to_replace_dct).values)

        with open(os.path.join(DATA_DIR, 'processed', 'ent2id.pkl'), 'wb') as fp:
            pickle.dump(self.entity2id, fp)

        with open(os.path.join(DATA_DIR, 'processed', 'rel2id.pkl'), 'wb') as fp:
            pickle.dump(self.relation2id, fp)

        self.write_to_npy(self.train_hrt, 'train.npy')
        self.write_to_npy(self.valid_hrt, 'valid.npy')
        self.write_to_npy(self.test_hrt, 'test.npy')

    def load_data(self, data_dir: str):
        with open(os.path.join(data_dir, 'processed', 'ent2id.pkl'), 'rb') as fp:
            self.entity2id = pickle.load(fp)

        with open(os.path.join(data_dir, 'processed', 'rel2id.pkl'), 'rb') as fp:
            self.relation2id = pickle.load(fp)

        self.num_entities = len(self.entity2id.keys())
        self.num_relations = len(self.relation2id.keys())

        self.train_hrt = self.load_from_npy('train.npy')
        self.valid_hrt = self.load_from_npy('valid.npy')
        self.test_hrt = self.load_from_npy('test.npy')

        self.entity_feat = self.load_from_npy('entity_features.npy')
        self.relation_feat = self.load_from_npy('relation_features.npy')

    def create_symbols_to_id(self):
        self.entity2id = defaultdict(int)
        for idx, wn_id in enumerate(self.entity_descs['code'].values):
            self.entity2id[wn_id] = idx

        self.relation2id = defaultdict(int)
        for idx, wn_id in enumerate(self.relation_descs.values):
            self.relation2id[wn_id] = idx

        # import IPython;IPython.embed()s

    def get_entity_features(self):
        print('Creating features using language model.')
        model = SentenceTransformer('stsb-distilroberta-base-v2')
        self.entity_feat = model.encode(self.entity_descs['description'].values)
        self.relation_feat = model.encode(self.relation_descs.values)


if __name__ == "__main__":
    wdn = ProcessWordNet()



