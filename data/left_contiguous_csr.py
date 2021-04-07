# Branch keshav

import numpy as np


class LeftContiguousCSR:
    def __init__(self, indptr: np.ndarray, degrees: np.ndarray, data: np.ndarray):
        self.indptr = indptr
        self.degrees = degrees
        self.data = data

    def __getitem__(self, i):
        start_ind = self.indptr[i]
        end_ind = start_ind + self.degrees[i]
        return self.data[start_ind:end_ind]

    def save(self, filepath):
        np.savez(filepath, indptr=self.indptr, degrees=self.degrees, data=self.data)

    @staticmethod
    def load(filepath):
        npzfile = np.load(filepath)
        return LeftContiguousCSR(npzfile['indptr'], npzfile['degrees'], npzfile['data'])

# uniform batch sampling can be done as follows

# self.data[
#   np.floor(
#     np.random.uniform(size=(batch.size, num_neighbors)) * degrees[batch].reshape(-1,1)
#   ).astype(np.int32)  +  indptr[batch].reshape(-1,1)
# ]

