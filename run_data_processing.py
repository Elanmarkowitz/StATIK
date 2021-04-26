import os

from data import data_processing

DATA_DIR = os.environ["DATA_DIR"] if "DATA_DIR" in os.environ else "/nas/home/elanmark/data"

if __name__ == "__main__":
    data_processing.process_data(DATA_DIR)
