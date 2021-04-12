# Branch keshav

import os
import pickle
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags
from ogb.lsc import WikiKG90MEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.data_loading import load_dataset, WikiKG90MProcessedDataset, Wiki90MValidationDataset
from model.kg_completion_gnn import KGCompletionGNN

FLAGS = flags.FLAGS
flags.DEFINE_string("root_data_dir", "/nas/home/elanmark/data", "Root data dir for installing the ogb dataset")
flags.DEFINE_integer("batch_size", 100, "Batch size. Number of triples.")
flags.DEFINE_integer("samples_per_node", 10, "Number of neighbors to sample for each entity in a query triple.")
flags.DEFINE_integer("embed_dim", 256, "Number of dimensions for hidden states.")
flags.DEFINE_integer("layers", 2, "Number of message passing and edge update layers for model.")
flags.DEFINE_integer("num_workers", 0, "Number of workers for the dataloader.")
flags.DEFINE_float("lr", 1e-2, "Learning rate for optimizer.")
flags.DEFINE_string("device", "cuda", "Device to use (cuda/cpu).")
flags.DEFINE_integer("print_freq", 1024, "How frequently to print learning statistics in number of iterations")
flags.DEFINE_integer("local_rank", 0, "How frequently to print learning statistics in number of iterations")
flags.DEFINE_integer("validate_every", 1024, "How many iterations to do between each single batch validation.")

DEBUGGING_MODEL = False


def prepare_batch_for_model(batch, dataset: WikiKG90MProcessedDataset, save_batch=False):
    ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, node_id_to_batch, queries, labels = batch
    if entity_feat is None:
        entity_feat = torch.from_numpy(dataset.entity_feat[entity_set]).float()
    relation_feat = torch.tensor(dataset.relation_feat).float()
    batch = ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels
    if save_batch:
        pickle.dump(batch, open('sample_batch.pkl', 'wb'))
    return batch


def move_batch_to_device(batch, device):
    ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels = batch
    ht_tensor_batch = ht_tensor_batch.to(device)
    r_tensor = r_tensor.to(device)
    entity_feat = entity_feat.to(device)
    relation_feat = relation_feat.to(device)
    queries = queries.to(device)
    labels = labels.to(device)
    batch = ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels
    return batch


def train(global_rank, local_rank):
    torch.cuda.set_device(local_rank)
    dataset = load_dataset(FLAGS.root_data_dir)
    sampler = DistributedSampler(dataset, rank=global_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers, sampler=sampler,
                              collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node))
    model = KGCompletionGNN(dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers)
    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    opt = optim.Adam(ddp_model.parameters(), lr=FLAGS.lr)
    moving_average_loss = torch.tensor(1.0)
    moving_average_acc = torch.tensor(0.5)
    moving_avg_rank = torch.tensor(10.0)
    for i, batch in enumerate(train_loader):
        ddp_model.train()
        batch = prepare_batch_for_model(batch, dataset)
        batch = move_batch_to_device(batch, local_rank)
        ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels = batch
        preds = ddp_model(ht_tensor_batch, r_tensor, entity_feat, relation_feat, queries)
        loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels.float())

        correct = torch.eq((preds > 0).long().flatten(), labels)
        score_1 = preds[labels == 1].detach().cpu().flatten()[0]
        score_0 = torch.topk(preds[labels == 0].detach().cpu().flatten().float()[:100], k=9).values
        rank = 1 + (score_1 < score_0).sum()
        moving_avg_rank = .9995 * moving_avg_rank + .0005 * rank.float()

        training_acc = correct.float().mean()
        moving_average_loss = .999 * moving_average_loss + 0.001 * loss.detach().cpu()
        moving_average_acc = .99 * moving_average_acc + 0.01 * training_acc.detach().cpu()

        if (i + 1) % FLAGS.print_freq == 0 and global_rank == 0:
            print(f"loss={loss.detach().cpu().numpy():.5f}, avg={moving_average_loss.numpy():.5f}, "
                  f"train_acc={training_acc.detach().cpu().numpy():.3f}, avg={moving_average_acc.numpy():.3f}, "
                  f"rank={rank} "
                  f"avg_rank={moving_avg_rank}")

            moving_average_loss = torch.tensor(1.0)
            moving_average_acc = torch.tensor(0.5)
            moving_avg_rank = torch.tensor(10.0)

        if (i + 1) % FLAGS.validate_every == 0 and global_rank == 0:
            ddp_model.eval()
            validate(dataset, ddp_model.module, single_itr=True)

        opt.zero_grad()
        loss.backward()
        opt.step()


# Validating only on global rank 0 for now
def validate(dataset: WikiKG90MProcessedDataset, model: KGCompletionGNN, single_itr=False):
    evaluator = WikiKG90MEvaluator()
    valid_dataset = Wiki90MValidationDataset(dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, num_workers=0, shuffle=True,
                                  collate_fn=valid_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))
    top_10s = []
    t_corrects = []
    with torch.no_grad():
        for i, (batch, t_correct_index) in enumerate(valid_dataloader):
            batch_preds = []
            for subbatch in batch:
                subbatch = prepare_batch_for_model(subbatch, valid_dataset.ds)
                subbatch = move_batch_to_device(subbatch, 0)
                ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels = subbatch
                preds = model(ht_tensor_batch, r_tensor, entity_feat, relation_feat, queries)
                batch_preds.append(preds)

            batch_preds = torch.cat(batch_preds, dim=1)
            t_pred_top10 = batch_preds.topk(10).indices
            t_pred_top10 = t_pred_top10.detach().cpu().numpy()
            batch_input_dict = {'h,r->t': {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}}
            top_10s.append(t_pred_top10)
            t_corrects.append(t_correct_index)
            # batch_result_dict = evaluator.eval(batch_input_dict)
            # print(batch_result_dict)
            if single_itr:
                break

    t_pred_top10 = np.concatenate(top_10s, axis=0)
    t_correct_index = np.concatenate(t_corrects, axis=0)
    input_dict = {'h,r->t': {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}}
    result_dict = evaluator.eval(input_dict)
    print(result_dict)


def test(dataset):
    pass


def main(argv):
    os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
    os.environ['GLOO_SOCKET_IFNAME'] = 'ib0'
    grank = int(os.environ['RANK'])
    ws = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    dist.init_process_group(backend=dist.Backend.NCCL,
                            init_method="tcp://{}:{}".format(master_addr, master_port), rank=grank, world_size=ws)
    train(grank, FLAGS.local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
