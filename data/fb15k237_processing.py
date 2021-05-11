import os
import numpy as np
from torch.utils.data import Dataset
import pickle
from sentence_transformers import SentenceTransformer


class FB15k237Dataset(Dataset):
    def __init__(self, root_data_dir):
        fb15k237path = root_data_dir + '/FB15k-237'
        if not os.path.isdir(fb15k237path):
            raise Exception('FB15k237 dataset not found in the directory {}'.format(root_data_dir))

        processed_dataset_dir = fb15k237path + '/processed'
        if not os.path.isdir(processed_dataset_dir):
            os.mkdir(processed_dataset_dir)
            self.train_hrt, self.val_hrt, self.test_hrt, self.entity_feat, self.num_entities, self.num_relations = process_dataset(fb15k237path)

            np.savez_compressed(processed_dataset_dir + '/fb15k237processed', train_hrt=self.train_hrt, val_hrt=self.val_hrt, test_hrt=self.test_hrt, entity_feat=self.entity_feat)
            stat_dict = {'num_entities': self.num_entities, 'num_relations': self.num_relations}
            with open(processed_dataset_dir + '/stat_dict.pkl', 'wb') as handle:
                pickle.dump(stat_dict, handle)

        else:
            loaded_array = np.load(processed_dataset_dir + '/fb15k237processed.npz')
            self.train_hrt = loaded_array['train_hrt']
            self.val_hrt = loaded_array['val_hrt']
            self.test_hrt = loaded_array['test_hrt']
            self.entity_feat = loaded_array['entity_feat']

            with open(processed_dataset_dir + '/stat_dict.pkl', 'rb') as handle:
                stat_dict = pickle.load(handle)

            self.num_entities = stat_dict['num_entities']
            self.num_relations = stat_dict['num_relations']


def process_dataset(dataset_path):
    def obtain_triples(file_path: str, entities2id_local: dict, relations2id_local: dict, entity2descriptions: dict, split='train'):
        triples = []
        with open(file_path) as f:
            for line in f:
                head, relation, tail = list(map(lambda x: x.strip('\n'), line.split('\t')))
                if head not in entity2descriptions or tail not in entity2descriptions:
                    continue
                if split == 'train':
                    if head not in entities2id_local:
                        entities2id_local[head] = len(entities2id_local)
                    if tail not in entities2id_local:
                        entities2id_local[tail] = len(entities2id_local)
                    if relation not in relations2id_local:
                        relations2id_local[relation] = len(relations2id_local)
                else:
                    if head not in entities2id_local or tail not in entities2id_local:
                        continue
                    if relation not in relations2id_local:
                        relations2id_local[relation] = len(relations2id_local)

                triples.append([entities2id_local[head], relations2id_local[relation], entities2id_local[tail]])

        return triples

    def obtain_feature_matrices(entityDescriptions: dict, entities2id_local: dict):
        entityId2description = {entities2id_local[entity]: entityDescriptions[entity] for entity, _ in entities2id_local.items()}
        sentences_list = [None] * len(entityId2description)

        for entityId, description in entityId2description.items():
            sentences_list[entityId] = description

        model = SentenceTransformer('stsb-roberta-base-v2')
        entity_feature_matrix = model.encode(sentences_list)

        return entity_feature_matrix

    def read_entity_descriptions(dataset_path):
        entity2description = {}
        with open(dataset_path + '/entitydescription.txt') as f:
            for line in f:
                entity, description = line.split('\t')
                cleaned_description = description.strip('\n').strip('@en').strip('"')
                entity2description[entity] = cleaned_description

        return entity2description

    entities2id_local = {}
    relations2id_local = {}
    entity2Descriptions = read_entity_descriptions(dataset_path)
    train_triples = obtain_triples(dataset_path + '/train.txt', entities2id_local, relations2id_local, entity2Descriptions, split='train')
    val_triples = obtain_triples(dataset_path + '/valid.txt', entities2id_local, relations2id_local, entity2Descriptions, split='val')
    test_triples = obtain_triples(dataset_path + '/test.txt', entities2id_local, relations2id_local, entity2Descriptions, split='test')
    entity_feature_matrix = obtain_feature_matrices(entity2Descriptions, entities2id_local)

    return np.asarray(train_triples), np.asarray(val_triples), np.asarray(test_triples), entity_feature_matrix, len(entities2id_local), len(relations2id_local)


if __name__ == '__main__':
    dataset = FB15k237Dataset('/home/keshav/datasets')
