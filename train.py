from absl import app, flags
import tqdm

import torch
from torch.utils.data import DataLoader

from data.data_loading import load_dataset

FLAGS = flags.FLAGS
flags.DEFINE_string("root_data_dir", "/nas/home/elanmark/data", "Root data dir for installing the ogb dataset")
flags.DEFINE_integer("batch_size", 100, "Batch size. Number of triples.")
flags.DEFINE_integer("samples_per_node", 10, "Number of neighbors to sample for each entity in a query triple.")


def main(argv):
    try:
        dataset = load_dataset(FLAGS.root_data_dir)
        train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True,
                                  collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node))
        for i, batch in tqdm.tqdm(enumerate(train_loader)):
            breakpoint()
            ht_tensor, r_tensor, entity_set, entity_feat, node_id_to_batch, queries, labels = batch
            relation_feat = dataset.relation_feat


    except:
        breakpoint()
        return main(argv)


if __name__ == "__main__":
    app.run(main)
