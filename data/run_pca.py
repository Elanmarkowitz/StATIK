from ogb.lsc import WikiKG90MDataset
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from tqdm import tqdm


def process_ent_features_pca(root_data_dir: str):
    dataset = WikiKG90MDataset(root=root_data_dir)
    ent_feat = dataset.entity_feat#:113]

    nc = 256
    chunk_size = 10000
 
    ipca = IncrementalPCA(n_components=nc)
    print('fitting pca ....')
    for i in tqdm(range(0, ent_feat.shape[0] // chunk_size+1)):
    	ipca.partial_fit(ent_feat[i*chunk_size : (i+1)*chunk_size])


    transformed_feat = np.memmap('/data/elanmark/wikikg90m_kddcup2021/processed/pca_entity_feat.npy', dtype='float32', mode='w+', shape=(ent_feat.shape[0], nc))
    print('transforming data ....')
    for i in tqdm(range(0, ent_feat.shape[0] // chunk_size+1)):
    	transformed_feat[i*chunk_size : (i+1)*chunk_size] = ipca.transform(ent_feat[i*chunk_size:(i+1)*chunk_size])[:, :nc]


    # import IPython; IPython.embed()

def process_rel_features_pca(root_data_dir: str):
    dataset = WikiKG90MDataset(root=root_data_dir)
    rel_feat = dataset.relation_feat

    nc = 256
    pca = PCA(n_components=nc)
    transformed_feat = np.memmap('/data/pca_relation_feat.npy', dtype='float32', mode='w+', shape=(rel_feat.shape[0], nc))
    transformed_feat = pca.fit_transform(rel_feat)


# process_features_pca('/data/elanmark/')
process_rel_features_pca('/data/elanmark/')