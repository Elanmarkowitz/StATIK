import os

from data.wordnet_processing_v2 import ProcessWordNet


ROOT_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/mehrnoom"

DATASET_INFO = {
        'dataset': 'Wikidata5M',
        'url': 'https://surfdrive.surf.nl/files/index.php/s/TEE96zweMxsoGmR/download',
        'train': 'ind-train.tsv',
        'test': 'ind-test.tsv',
        'dev': 'ind-dev.tsv',
        'ent_desc': 'entity2textlong.txt',
        'ent_desc2': 'entity2text.txt',
        'rel_desc': 'relation2text.txt'
    }


class ProcessWikidata5M(ProcessWordNet):
    def __init__(self, root_data_dir=None, dataset_info=None):
        dataset_info = DATASET_INFO if dataset_info is None else dataset_info
        super(ProcessWikidata5M, self).__init__(root_data_dir=root_data_dir, dataset_info=dataset_info)
