import math
import random

from absl import app, flags
import os
import pickle
import numpy as np

import setproctitle

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from ogb.lsc import WikiKG90MEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from data.data_loading import load_dataset, WikiKG90MProcessedDataset, Wiki90MValidationDataset
from model.kg_completion_gnn import KGCompletionGNN
from utils import load_model, load_opt_checkpoint, save_checkpoint, save_model

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
flags.DEFINE_bool("edge_attention", False, "Whether or not to attend to edges during ")
flags.DEFINE_bool("relation_scoring", False, "Whether or not to learn pairwise relation importance")
flags.DEFINE_integer("validation_batches", 1000, "Number of batches to do for each validation check.")
flags.DEFINE_integer("valid_batch_size", 1, "Batch size for validation (does all t_candidates at once).")
flags.DEFINE_integer("epochs", 1, "Num epochs to train for")
flags.DEFINE_bool('inference_only', False, 'Whether or not to do a complete inference run across the validation set')
flags.DEFINE_string('model_path_depr', None, 'DEPRECATED: Path where the model is saved')
flags.DEFINE_string('model_path', None, 'Path where the model is saved (inference only)')
flags.DEFINE_bool("distributed", True, "Indicate whether to use distributed training.")
flags.DEFINE_string("checkpoint", None, "Resume training from checkpoint file in checkpoints directory.")
flags.DEFINE_string("name", "run01", "A name to use for saving the run.")

CHECKPOINT_DIR = "checkpoints"


def prepare_batch_for_model(batch, dataset: WikiKG90MProcessedDataset, save_batch=False):
    ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
    if entity_feat is None:
        entity_feat = torch.from_numpy(dataset.entity_feat[entity_set]).float()

    batch = ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
    if save_batch:
        pickle.dump(batch, open('sample_batch.pkl', 'wb'))
    return batch


def move_batch_to_device(batch, device):
    ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
    ht_tensor = ht_tensor.to(device)
    r_tensor = r_tensor.to(device)
    entity_feat = entity_feat.to(device)
    queries = queries.to(device)
    labels = labels.to(device)
    r_queries = r_queries.to(device)
    r_relatives = r_relatives.to(device)
    h_or_t_sample = h_or_t_sample.to(device)
    batch = ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
    return batch


def train(global_rank, local_rank, world):
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataset = load_dataset(FLAGS.root_data_dir)
    train_sampler = DistributedSampler(dataset, rank=global_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers, sampler=train_sampler,
                              collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node, sample_negs=1))

    valid_dataset = Wiki90MValidationDataset(dataset)
    valid_sampler = DistributedSampler(valid_dataset, rank=global_rank, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers, sampler=valid_sampler,
                                  drop_last=True, collate_fn=valid_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))

    if FLAGS.checkpoint is not None:
        model = load_model(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint))
    else:
        model = KGCompletionGNN(dataset.num_relations, dataset.relation_feat, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers,
                                    edge_attention=FLAGS.edge_attention,
                                    relation_scoring=FLAGS.relation_scoring)
    model.to(local_rank)

    ddp_model = DDP(model, device_ids=[local_rank], process_group=world, find_unused_parameters=True)
    opt = optim.Adam(ddp_model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, gamma=0.5,
                                               milestones=[math.floor(len(train_loader) / 4),
                                                           math.floor(len(train_loader) / 2), len(train_loader)])
    start_epoch = 0
    if FLAGS.checkpoint:
        start_epoch = load_opt_checkpoint(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint), opt, scheduler)

    moving_average_loss = torch.tensor(1.0, device=local_rank)
    moving_average_acc = torch.tensor(0.5, device=local_rank)
    moving_avg_rank = torch.tensor(10.0, device=local_rank)
    max_mrr = 0

    for epoch in range(start_epoch, FLAGS.epochs):
        for i, batch in enumerate(tqdm(train_loader)):
            ddp_model.train()
            batch = prepare_batch_for_model(batch, dataset)
            batch = move_batch_to_device(batch, local_rank)
            ht_tensor, r_tensor, entity_set, entity_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
            preds = ddp_model(ht_tensor, r_tensor, entity_feat, queries)

            loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels.float())

            correct = torch.eq((preds > 0).long().flatten(), labels)
            score_1 = preds[labels == 1].detach().cpu().flatten()[0]
            score_0 = torch.topk(preds[labels == 0].detach().cpu().flatten().float()[:100], k=9).values
            rank = 1 + torch.less(score_1, score_0).sum()
            moving_avg_rank = .9995 * moving_avg_rank + .0005 * rank.float()
            training_acc = correct.float().mean()

            moving_average_loss = .999 * moving_average_loss + 0.001 * loss.detach()
            moving_average_acc = .99 * moving_average_acc + 0.01 * training_acc.detach()

            if (i + 1) % FLAGS.print_freq == 0:
                dist.all_reduce(moving_average_loss, group=world)
                dist.all_reduce(moving_average_acc, group=world)
                dist.all_reduce(moving_avg_rank, group=world)

                moving_average_loss /= dist.get_world_size()
                moving_average_acc /= dist.get_world_size()
                moving_avg_rank /= dist.get_world_size()

                if global_rank == 0:
                    print(f"Iteration={i}/{len(train_loader)}, "
                          f"Moving Avg Loss={moving_average_loss.cpu().numpy():.5f}, "
                          f"Moving Avg Train Acc={moving_average_acc.cpu().numpy():.3f}, "
                          f"Moving Avg Rank={moving_avg_rank.cpu().numpy()}")

            if (i + 1) % FLAGS.validate_every == 0:
                ddp_model.eval()
                result = validate(valid_dataset, valid_dataloader, ddp_model, global_rank, local_rank, num_batches=FLAGS.validation_batches,
                                  world=world)
                if global_rank == 0:
                    mrr = result['mrr']
                    if mrr > max_mrr:
                        max_mrr = mrr
                        save_model(ddp_model.module, os.path.join(CHECKPOINT_DIR, f'{FLAGS.name}_best_model.pkl'))

                    print('Current MRR = {}, Best MRR = {}'.format(mrr, max_mrr))

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # TODO: remove this!
            if (i+1) % 10 == 0:
                break

        save_checkpoint(ddp_model.module, epoch+1, opt, scheduler, os.path.join(CHECKPOINT_DIR, f"{FLAGS.name}_e{epoch}.pkl"))


def train_inner(model, train_loader, opt, dataset, device, print_output=True):
    moving_average_loss = torch.tensor(1.0)
    moving_average_acc = torch.tensor(0.5)
    moving_avg_rank = torch.tensor(10.0)
    for i, batch in enumerate(tqdm.tqdm(train_loader)):
        model.train()
        batch = prepare_batch_for_model(batch, dataset)
        batch = move_batch_to_device(batch, device)
        ht_tensor, r_tensor, entity_set, entity_feat, relation_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
        preds = model(ht_tensor, r_tensor, entity_feat, relation_feat, queries)
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


def inference_only(global_rank, local_rank, world):
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataset = load_dataset(FLAGS.root_data_dir)
    valid_dataset = Wiki90MValidationDataset(dataset)
    valid_sampler = DistributedSampler(valid_dataset, rank=global_rank, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers, sampler=valid_sampler,
                                  drop_last=True, collate_fn=valid_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))
    model = KGCompletionGNN(dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers, edge_attention=FLAGS.edge_attention,
                            relation_scoring=FLAGS.relation_scoring)

    if global_rank == 0:
        assert FLAGS.model_path_depr is not None or FLAGS.model_path is not None, 'Must be supplied with model to do inference.'
        if FLAGS.model_path_depr is not None:
            model.load_state_dict(torch.load(FLAGS))
        elif FLAGS.model_path is not None:
            model = load_model(FLAGS.model_path)


    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    ddp_model.eval()
    result = validate(valid_dataset, valid_dataloader, ddp_model, global_rank, local_rank, FLAGS.validation_batches, world)
    if global_rank == 0:
        mrr = result['mrr']
        print('Validation MRR = {}'.format(mrr))


def validate(valid_dataset: Dataset, valid_dataloader: DataLoader, model, global_rank: int, local_rank: int, num_batches: int = None,
             world=None):
    evaluator = WikiKG90MEvaluator()

    top_10s = []
    t_corrects = []
    with torch.no_grad():
        for i, (batch, t_correct_index) in enumerate(tqdm(valid_dataloader)):
            batch = prepare_batch_for_model(batch, valid_dataset.ds)
            batch = move_batch_to_device(batch, local_rank)
            ht_tensor, r_tensor, entity_set, entity_feat, queries, _, r_queries, r_relatives, h_or_t_sample = batch
            preds = model(ht_tensor, r_tensor, entity_feat, queries)
            preds = preds.reshape(FLAGS.valid_batch_size, 1001)
            t_pred_top10 = preds.topk(10).indices
            t_pred_top10 = t_pred_top10.detach()
            t_correct_index = torch.tensor(t_correct_index, device=local_rank)
            top_10s.append(t_pred_top10)
            t_corrects.append(t_correct_index)
            if num_batches and num_batches == (i + 1):
                break

    t_pred_top10 = torch.cat(top_10s, dim=0)
    t_correct_index = torch.cat(t_corrects, dim=0)
    aggregated_top10_preds, aggregated_correct_indices = gather_results(t_pred_top10, t_correct_index, global_rank, world)
    if global_rank == 0:
        input_dict = {'h,r->t': {'t_pred_top10': aggregated_top10_preds.cpu().numpy(), 't_correct_index': aggregated_correct_indices.cpu().numpy()}}
        result_dict = evaluator.eval(input_dict)
        return result_dict
    else:
        return None


def gather_results(t_pred_top10: torch.Tensor, t_correct_index: torch.Tensor, global_rank, world):
    gather_list_pred = [torch.empty(t_pred_top10.shape, dtype=t_pred_top10.dtype, device=global_rank) for _ in range(dist.get_world_size())]
    gather_list_correct = [torch.empty(t_correct_index.shape, dtype=t_correct_index.dtype, device=global_rank) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list_pred, t_pred_top10, group=world)
    dist.all_gather(gather_list_correct, t_correct_index, group=world)
    return torch.cat(gather_list_pred, dim=0), torch.cat(gather_list_correct, dim=0)


def main(argv):
    if FLAGS.distributed:
        grank = int(os.environ['RANK'])
        ws = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        dist.init_process_group(backend=dist.Backend.NCCL,
                                init_method="tcp://{}:{}".format(master_addr, master_port), rank=grank, world_size=ws)

        setproctitle.setproctitle("KGCompletionTrainer:{}".format(grank))
        world = dist.new_group([i for i in range(ws)], backend=dist.Backend.NCCL)

        if FLAGS.edge_attention and FLAGS.relation_scoring:
            raise Exception("Only one of relation scoring or edge attention can be enabled!")
        if FLAGS.inference_only:
            inference_only(grank, FLAGS.local_rank, FLAGS.model_path, world)
        else:
            train(grank, FLAGS.local_rank, world)
        dist.destroy_process_group()
    else:
        train_single(FLAGS.device)


if __name__ == "__main__":
    app.run(main)
