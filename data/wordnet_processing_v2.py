import tarfile

import pandas as pd
import numpy as np
import os
import pickle

from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, FeatureExtractionPipeline

import urllib.request


ROOT_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/mehrnoom"

DATASET_INFO = {
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
        self.entity_text = None
        self.relation_text = None

        self.dataset_info = DATASET_INFO if dataset_info is None else dataset_info
        if root_data_dir is None:
            self.data_dir = os.path.join(ROOT_DIR, self.dataset_info['dataset'])
        else:
            self.data_dir = os.path.join(root_data_dir, self.dataset_info['dataset'])

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

    def read_triples(self, filename) -> pd.DataFrame:
        triples = pd.read_csv(os.path.join(self.data_dir, filename), names=['h', 'r', 't'], sep='\t')
        return triples

    @staticmethod
    def get_first_meaningful_sentence(desc):
        sentences = desc.split('.')
        cumulative_words = 0
        for i in range(len(sentences)):
            cumulative_words += len(sentences[0].split(' '))
            if cumulative_words > 5:
                return '.'.join(sentences[:i + 1])
        return desc

    @staticmethod
    def get_first_n_words(desc, n=32):
        words = desc.split(' ')
        return ' '.join(words[:n])

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
        self.entity_text = self.entity_descs['description'].values
        self.relation_text = self.relation_descs['description'].values
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

        self.train_hrt = np.asarray(self.replace_hrt(self.train_hrt, to_replace_dct).values, dtype=np.int)
        self.valid_hrt = np.asarray(self.replace_hrt(self.valid_hrt, to_replace_dct).values, dtype=np.int)
        self.test_hrt = np.asarray(self.replace_hrt(self.test_hrt, to_replace_dct).values, dtype=np.int)

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

        entity_descs, relation_descs = self.read_descriptions()
        self.entity_text = entity_descs['description'].values
        self.relation_text = relation_descs['description'].values

    def create_symbols_to_id(self):
        # self.entity2id = defaultdict(int)
        # for idx, wn_id in enumerate(self.entity_descs['code'].values):
        #     self.entity2id[wn_id] = idx

        self.entity2id = defaultdict(int)
        train_ent = set(self.train_hrt['h'].values)
        train_ent.update(set(self.train_hrt['t'].values))
        idx = 0
        for wn_id in train_ent:
            if wn_id not in self.entity2id:
                self.entity2id[wn_id] = idx
                idx += 1

        dev_ent = set(self.valid_hrt['h'].values)
        dev_ent.update(self.valid_hrt['t'].values)
        for wn_id in dev_ent:
            if wn_id not in self.entity2id:
                self.entity2id[wn_id] = idx
                idx += 1

        test_ent = set(self.test_hrt['h'].values)
        test_ent.update(self.test_hrt['t'].values)
        for wn_id in test_ent:
            if wn_id not in self.entity2id:
                self.entity2id[wn_id] = idx
                idx += 1



        self.relation2id = defaultdict(int)
        for idx, wn_id in enumerate(self.relation_descs['code'].values):
            self.relation2id[wn_id] = idx

        # import IPython;IPython.embed()s

    def get_entity_features(self):
        print('Creating features using language model.')
        # if self.dataset_info['dataset'] == 'FB15k-237':
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        #     model = BertModel.from_pretrained('bert-base-cased')
        #     pipeline = FeatureExtractionPipeline(model, tokenizer, device=0)
        #     self.entity_feat = np.array([np.array(pipeline(e))[0,0,:].flatten()
        #                                  for e in self.entity_descs['description'].apply(ProcessWordNet.get_first_n_words).values])
        #     self.relation_feat = np.array([np.array(pipeline(e))[0,0,:].flatten()
        #                                    for e in self.relation_descs['description'].apply(ProcessWordNet.get_first_n_words).values])
        # else:
        model = SentenceTransformer('stsb-distilroberta-base-v2')
        self.entity_feat = model.encode(self.entity_descs['description'].apply(ProcessWordNet.get_first_n_words).values)
        self.relation_feat = model.encode(self.relation_descs['description'].apply(ProcessWordNet.get_first_n_words).values)

    @staticmethod
    def replace_hrt(hrt: pd.DataFrame, map_dict):
        for key in map_dict.keys():
            hrt[key] = hrt[key].map(map_dict[key])
        return hrt


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



