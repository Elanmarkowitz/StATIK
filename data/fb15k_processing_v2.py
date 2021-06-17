import os

from data.wordnet_processing_v2 import ProcessWordNet


ROOT_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/mehrnoom"

DATASET_INFO = {
        'dataset': 'FB15k-237',
        'url': 'https://surfdrive.surf.nl/files/index.php/s/rGqLTDXRFLPJYg7/download',
        'train': 'ind-train.tsv',
        'test': 'ind-test.tsv',
        'dev': 'ind-dev.tsv',
        'ent_desc': 'entity2textlong.txt',
        'rel_desc': 'relation2text.txt'
    }


class ProcessFreebase(ProcessWordNet):
    def __init__(self, root_data_dir=None, dataset_info=None):
        dataset_info = DATASET_INFO if dataset_info is None else dataset_info
        super(ProcessFreebase, self).__init__(root_data_dir=root_data_dir, dataset_info=dataset_info)
