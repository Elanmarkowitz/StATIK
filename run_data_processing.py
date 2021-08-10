import os

from data import data_processing

DATA_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/data/elanmark"
DATASET = os.environ["DATASET"] if "DATASET" in os.environ else "wikikg90m_kddcup2021"


transfer_setting = (DATASET == 'Wikidata5M')

if __name__ == "__main__":
    data_processing.process_data(DATA_DIR, DATASET, transfer_setting=transfer_setting)
