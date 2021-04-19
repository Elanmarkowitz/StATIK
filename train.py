from absl import app, flags
import os
import pickle
import numpy as np
import tqdm

import setproctitle
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
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
flags.DEFINE_integer("local_rank", 0, "How frequently to print learning statistics in number of iterations")  # TODO: fix help message
flags.DEFINE_integer("validate_every", 1024, "How many iterations to do between each single batch validation.")
flags.DEFINE_integer("validation_batches", 1000, "Number of batches to do for each validation check.")
flags.DEFINE_integer("valid_batch_size", 1, "Batch size for validation (does all t_candidates at once).")
flags.DEFINE_bool("parameterized_sampling", True, "Should parameterized sampling be used?")
flags.DEFINE_bool("distributed", True, "Indicate whether to use distributed training.")

DEBUGGING_MODEL = False


def prepare_batch_for_model(batch, dataset: WikiKG90MProcessedDataset, save_batch=False):
    ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, p_selections = batch
    if entity_feat is None:
        entity_feat = torch.from_numpy(dataset.entity_feat[entity_set]).float()
    relation_feat = torch.tensor(dataset.relation_feat).float()
    batch = ht_tensor, r_tensor, entity_set, entity_feat, relation_feat, queries, labels, p_selections
    if save_batch:
        pickle.dump(batch, open('sample_batch.pkl', 'wb'))
    return batch


def move_batch_to_device(batch, device):
    ht_tensor, r_tensor, entity_set, entity_feat, relation_feat, queries, labels, p_selections = batch
    ht_tensor = ht_tensor.to(device)
    r_tensor = r_tensor.to(device)
    entity_feat = entity_feat.to(device)
    relation_feat = relation_feat.to(device)
    queries = queries.to(device)
    labels = labels.to(device)
    if p_selections is not None:
        p_selections = p_selections.to(device)
    batch = ht_tensor, r_tensor, entity_set, entity_feat, relation_feat, queries, labels, p_selections
    return batch


def train_inner(model, train_loader, opt, dataset, device, print_output=True):
    moving_average_loss = torch.tensor(1.0)
    moving_average_acc = torch.tensor(0.5)
    moving_avg_rank = torch.tensor(10.0)
    for i, batch in enumerate(tqdm.tqdm(train_loader)):
        model.train()
        batch = prepare_batch_for_model(batch, dataset)
        batch = move_batch_to_device(batch, device)
        ht_tensor, r_tensor, entity_set, entity_feat, relation_feat, queries, labels, p_selections = batch
        preds = model(ht_tensor, r_tensor, entity_feat, relation_feat, p_selections, queries)
        loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels.float())

        correct = torch.eq((preds > 0).long().flatten(), labels)
        score_1 = preds[labels == 1].detach().cpu().flatten()[0]
        score_0 = torch.topk(preds[labels == 0].detach().cpu().flatten().float()[:100], k=9).values
        rank = 1 + torch.less(score_1, score_0).sum()
        moving_avg_rank = .9995 * moving_avg_rank + .0005 * rank.float()

        training_acc = correct.float().mean()
        moving_average_loss = .999 * moving_average_loss + 0.001 * loss.detach().cpu()
        moving_average_acc = .99 * moving_average_acc + 0.01 * training_acc.detach().cpu()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i + 1) % FLAGS.print_freq == 0 and print_output:
            print(f"loss={loss.detach().cpu().numpy():.5f}, avg={moving_average_loss.numpy():.5f}, "
                  f"train_acc={training_acc.detach().cpu().numpy():.3f}, avg={moving_average_acc.numpy():.3f}, "
                  f"rank={rank} "
                  f"avg_rank={moving_avg_rank}")

        if (i + 1) % FLAGS.validate_every == 0 and print_output:
            model.eval()
            module = model.module if FLAGS.distributed else model
            validate(dataset, module, num_batches=FLAGS.validation_batches)


def train_single(device):
    dataset = load_dataset(FLAGS.root_data_dir)
    train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
                              collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node, sample_negs=1))
    model = KGCompletionGNN(dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr)
    train_inner(model, train_loader, opt, dataset, device)


def train_distributed(global_rank, local_rank):
    torch.cuda.set_device(local_rank)
    dataset = load_dataset(FLAGS.root_data_dir)
    sampler = DistributedSampler(dataset, rank=global_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, sampler=sampler,
                              collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node, sample_negs=1))
    model = KGCompletionGNN(dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers)
    model.to(local_rank)
    if FLAGS.distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr)
    train_inner(model, train_loader, opt, dataset, local_rank, print_output=(global_rank==0))


# Validating only on global rank 0 for now
def validate(dataset: WikiKG90MProcessedDataset, model: torch.nn.Module, num_batches: int = None):
    evaluator = WikiKG90MEvaluator()
    valid_dataset = Wiki90MValidationDataset(dataset)
    VALID_BATCH_SIZE = FLAGS.valid_batch_size
    valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=0, shuffle=True,
                                  collate_fn=valid_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))
    top_10s = []
    t_corrects = []
    with torch.no_grad():
        for i, (batch, t_correct_index) in enumerate(valid_dataloader):
            batch = prepare_batch_for_model(batch, valid_dataset.ds)
            batch = move_batch_to_device(batch, 0)  # TODO: This probably needs to be changed for DDP
            ht_tensor, r_tensor, entity_set, entity_feat, relation_feat, queries, _, p_selections = batch
            preds = model(ht_tensor, r_tensor, entity_feat, relation_feat, p_selections, queries)
            preds = preds.reshape(VALID_BATCH_SIZE, 1001)
            t_pred_top10 = preds.topk(10).indices
            t_pred_top10 = t_pred_top10.detach().cpu().numpy()
            top_10s.append(t_pred_top10)
            t_corrects.append(t_correct_index)
            if num_batches and num_batches == (i + 1):
                break
    t_pred_top10 = np.concatenate(top_10s, axis=0)
    t_correct_index = np.concatenate(t_corrects, axis=0)
    input_dict = {'h,r->t': {'t_pred_top10': t_pred_top10, 't_correct_index': t_correct_index}}
    result_dict = evaluator.eval(input_dict)
    print(result_dict)


def test(dataset):
    pass


def main(argv):
    if FLAGS.distributed:
        grank = int(os.environ['RANK'])
        ws = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        dist.init_process_group(backend=dist.Backend.NCCL,
                                init_method="tcp://{}:{}".format(master_addr, master_port), rank=grank, world_size=ws)

        setproctitle.setproctitle("KGCompletionTrainer:{}".format(grank))
        train_distributed(grank, FLAGS.local_rank)
        dist.destroy_process_group()
    else:
        train_single(FLAGS.device)


if __name__ == "__main__":
    app.run(main)
