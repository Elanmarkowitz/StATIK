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
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset, ConcatDataset
from tqdm import tqdm

from attributed_eval import AttributedEvaluator, convert_stats_to_percentiles
from data.data_loading import load_dataset, KGProcessedDataset, KGValidationDataset, KGEvaluationDataset, KGTestDataset
from evaluation import compute_eval_stats
from model.kg_completion_gnn import KGCompletionGNN
from utils import load_model, load_opt_checkpoint, save_checkpoint, save_model

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "wikikg90m_kddcup2021", "Specifies dataset from [wikikg90m_kddcup2021, wordnet-mlj12]")
flags.DEFINE_string("root_data_dir", "/nas/home/elanmark/data", "Root data dir for installing the ogb dataset")

flags.DEFINE_bool("distributed", True, "Indicate whether to use distributed training.")
flags.DEFINE_integer("num_workers", 0, "Number of workers for the dataloader.")
flags.DEFINE_integer("local_rank", 0, "How frequently to print learning statistics in number of iterations")  # TODO: Change description here
flags.DEFINE_integer("print_freq", 1024, "How frequently to print learning statistics in number of iterations")
flags.DEFINE_string("device", "cuda", "Device to use (cuda/cpu).")
flags.DEFINE_string("checkpoint", None, "Resume training from checkpoint file in checkpoints directory.")
flags.DEFINE_string("name", "run01", "A name to use for saving the run.")

flags.DEFINE_integer("batch_size", 100, "Batch size. Number of triples.")
flags.DEFINE_integer("neg_samples", 1, "Number of neg samples per positive tail during training.")
flags.DEFINE_integer("samples_per_node", 10, "Number of neighbors to sample for each entity in a query triple.")
flags.DEFINE_integer("embed_dim", 256, "Number of dimensions for hidden states.")
flags.DEFINE_integer("layers", 2, "Number of message passing and edge update layers for model.")
flags.DEFINE_float("lr", 1e-2, "Learning rate for optimizer.")
flags.DEFINE_integer("validate_every", 1024, "How many iterations to do between each single batch validation.")
flags.DEFINE_integer("validation_batches", 1000, "Number of batches to do for each validation check.")
flags.DEFINE_integer("valid_batch_size", 1, "Batch size for validation (does all t_candidates at once).")
flags.DEFINE_integer("epochs", 1, "Num epochs to train for")
flags.DEFINE_float("margin", 1.0, "Margin in transE loss")
flags.DEFINE_string("decoder", "MLP+TransE", "Choose decoder from [MLP, TransE, MLP+TransE]")

flags.DEFINE_bool('validation_only', False, 'Whether or not to do a complete inference run across the validation set')
flags.DEFINE_bool('test_only', False, 'Whether or not to do a complete inference run across the test set')
flags.DEFINE_string('model_path_depr', None, 'DEPRECATED: Path where the model is saved')
flags.DEFINE_string('model_path', None, 'Path where the model is saved (inference only)')
flags.DEFINE_string("test_save_dir", "test_submissions", "Directory to save test results file in.")
flags.DEFINE_bool("validation_attribution", False, "Whether to perform validation attribution on full validation runs")

flags.DEFINE_bool("predict_heads", True, "Whether to predict heads and sample heads at training.")

CHECKPOINT_DIR = "checkpoints"


def prepare_batch_for_model(batch, dataset: KGProcessedDataset, save_batch=False):
    ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
    if entity_feat is None:
        entity_feat = torch.from_numpy(dataset.entity_feat[entity_set]).float()

    batch = ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
    if save_batch:
        pickle.dump(batch, open('sample_batch.pkl', 'wb'))
    return batch


def move_batch_to_device(batch, device):
    ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
    ht_tensor = ht_tensor.to(device)
    r_tensor = r_tensor.to(device)
    entity_feat = entity_feat.to(device)
    indeg_feat = indeg_feat.to(device)
    outdeg_feat = outdeg_feat.to(device)
    queries = queries.to(device)
    labels = labels.to(device)
    r_queries = r_queries.to(device)
    r_relatives = r_relatives.to(device)
    h_or_t_sample = h_or_t_sample.to(device)
    batch = ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample
    return batch


def train(global_rank, local_rank, world):
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataset = load_dataset(FLAGS.root_data_dir, FLAGS.dataset)
    train_sampler = DistributedSampler(dataset, rank=global_rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers, sampler=train_sampler,
                              collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node,
                                                                sample_negs=FLAGS.neg_samples,
                                                                neg_heads=FLAGS.predict_heads))

    valid_dataset = KGValidationDataset(dataset)
    valid_sampler = DistributedSampler(valid_dataset, rank=global_rank, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers, sampler=valid_sampler,
                                  drop_last=True, collate_fn=valid_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))

    if FLAGS.predict_heads:
        valid_dataset_head = KGValidationDataset(dataset, head_prediction=True)
        valid_sampler_head = DistributedSampler(valid_dataset_head, rank=global_rank, shuffle=False)
        valid_dataloader_head = DataLoader(valid_dataset_head, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                           sampler=valid_sampler,
                                           drop_last=True, collate_fn=valid_dataset_head.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))


    if FLAGS.checkpoint is not None:
        model = load_model(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint), ignore_state_dict=(global_rank != 0))
    else:
        model = KGCompletionGNN(dataset.relation_feat, dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers, decoder=FLAGS.decoder)

    model.to(local_rank)

    ddp_model = DDP(model, device_ids=[local_rank], process_group=world, find_unused_parameters=True)
    loss_fn = model.get_loss_fn(margin=FLAGS.margin)
    opt = optim.Adam(ddp_model.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(opt,
                                               milestones=[len(train_loader), 2 * len(train_loader), 3 * len(train_loader)],
                                               gamma=0.5)

    start_epoch = 0
    if FLAGS.checkpoint:
        start_epoch = load_opt_checkpoint(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint), opt, scheduler)

    moving_average_loss = torch.tensor(1.0, device=local_rank)
    max_mrr = 0

    for epoch in range(start_epoch, FLAGS.epochs):
        if global_rank == 0:
            print(f'Epoch {epoch}')
        for i, batch in enumerate(tqdm(train_loader)):
            ddp_model.train()
            batch = prepare_batch_for_model(batch, dataset)
            batch = move_batch_to_device(batch, local_rank)
            ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, labels, r_queries, r_relatives, h_or_t_sample = batch
            scores = ddp_model(ht_tensor, r_tensor, r_queries, entity_feat, r_relatives, h_or_t_sample, queries)

            loss = loss_fn(scores, labels.float())

            moving_average_loss = .999 * moving_average_loss + 0.001 * loss.detach()

            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()

            if (i + 1) % FLAGS.print_freq == 0:
                dist.all_reduce(moving_average_loss, group=world)
                moving_average_loss /= dist.get_world_size()

                if global_rank == 0:
                    print(f"Iteration={i}/{len(train_loader)}, "
                          f"Moving Avg Loss={moving_average_loss.cpu().numpy():.5f}")

        if (epoch + 1) % FLAGS.validate_every == 0:
            ddp_model.eval()
            gather_sizes = [FLAGS.valid_batch_size * FLAGS.validation_batches] * world.size()
            result = validate(valid_dataset, valid_dataloader, ddp_model, global_rank, local_rank, gather_sizes, num_batches=FLAGS.validation_batches,
                              world=world)
            if FLAGS.predict_heads:
                result2 = validate(valid_dataset_head, valid_dataloader_head, ddp_model, global_rank, local_rank,
                                   gather_sizes, num_batches=FLAGS.validation_batches, world=world)
                if global_rank == 0:
                    result['mrr'] = 0.5*result['mrr'] + 0.5*result2['mrr']
            if global_rank == 0:
                mrr = result['mrr']
                if mrr > max_mrr:
                    max_mrr = mrr
                    save_model(ddp_model.module, os.path.join(CHECKPOINT_DIR, f'{FLAGS.name}_best_model.pkl'))


                print('Current MRR = {}, Best MRR = {}'.format(mrr, max_mrr))

        if global_rank == 0:
            save_checkpoint(ddp_model.module, epoch + 1, opt, scheduler, os.path.join(CHECKPOINT_DIR, f"{FLAGS.name}_e{epoch}.pkl"))


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
    dataset = load_dataset(FLAGS.root_data_dir, FLAGS.dataset)
    train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
                              collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node, sample_negs=1))
    model = KGCompletionGNN(dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr)
    train_inner(model, train_loader, opt, dataset, device)


def inference_only(global_rank, local_rank, world):
    assert FLAGS.test_only or FLAGS.validation_only, "Must run validation set or test set if running inference."
    assert not (FLAGS.test_only and FLAGS.validation_only), "Can only run on one of test or validation when doing inference."
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    base_dataset = load_dataset(FLAGS.root_data_dir, FLAGS.dataset)
    if FLAGS.validation_only:
        eval_dataset = KGValidationDataset(base_dataset, head_prediction=FLAGS.predict_heads)
    else:
        eval_dataset = KGTestDataset(base_dataset, head_prediction=FLAGS.predict_heads)

    num_ranks = world.size()
    idxs_per_rank = math.ceil(len(eval_dataset) / num_ranks)
    start_idx = global_rank * idxs_per_rank
    end_idx = (global_rank + 1) * idxs_per_rank if ((global_rank + 1) * idxs_per_rank <= len(eval_dataset)) else len(eval_dataset)
    rank_idxs = torch.arange(start_idx, end_idx, dtype=torch.long).tolist()
    print(f"Global rank {global_rank} processing dataset from {rank_idxs[0]} through {rank_idxs[-1]}")

    subset = Subset(eval_dataset, rank_idxs)
    eval_dataloader = DataLoader(subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                 collate_fn=eval_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))

    assert FLAGS.model_path_depr is not None or FLAGS.model_path is not None, 'Must be supplied with model to do inference.'
    if FLAGS.model_path_depr is not None:
        model = KGCompletionGNN(base_dataset.relation_feat, base_dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers, decoder=FLAGS.decoder)
        if global_rank == 0:
            model.load_state_dict(torch.load(FLAGS.model_path_depr))
    elif FLAGS.model_path is not None:
        model = load_model(FLAGS.model_path, ignore_state_dict=(global_rank != 0))
    else:
        raise Exception('Must be supplied with model to do inference.')
    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], process_group=world)
    ddp_model.eval()

    if FLAGS.test_only or FLAGS.validation_batches < 0:
        FLAGS.validation_batches = None
        gather_sizes = [math.ceil(len(eval_dataset) / num_ranks)] * (num_ranks - 1)
        last_size = len(eval_dataset) - sum(gather_sizes)
        gather_sizes.append(last_size)
    else:
        gather_sizes = [FLAGS.valid_batch_size * FLAGS.validation_batches] * num_ranks

    if FLAGS.validation_only:
        result = validate(eval_dataset, eval_dataloader, ddp_model, global_rank, local_rank, gather_sizes, FLAGS.validation_batches, world)
        if global_rank == 0:
            print('Validation ' + ' '.join([f'{k}={result[k]}' for k in result.keys()]))


    else:
        test(eval_dataset, eval_dataloader, ddp_model, global_rank, local_rank, gather_sizes, None, world)


def run_inference(dataset: KGEvaluationDataset, dataloader: DataLoader, model, global_rank: int, local_rank: int,
                  gather_sizes: list, num_batches: int = None, world=None, use_full_preds=False):
    top_10s = []
    t_corrects = []
    full_preds = []
    filter_masks = []
    with torch.no_grad():
        for i, (subbatches, t_correct_index, t_filter_mask) in enumerate(tqdm(dataloader)):
            preds = []
            for subbatch in subbatches:
                subbatch = prepare_batch_for_model(subbatch, dataset.ds)
                subbatch = move_batch_to_device(subbatch, local_rank)
                ht_tensor, r_tensor, entity_set, entity_feat, indeg_feat, outdeg_feat, queries, _, r_queries, r_relatives, h_or_t_sample = subbatch
                subbatch_preds = model(ht_tensor, r_tensor, r_queries, entity_feat, r_relatives, h_or_t_sample, queries)
                subbatch_preds = subbatch_preds.reshape(t_correct_index.shape[0], -1)  # TODO: inferring number of candidates, check that this is right.
                preds.append(subbatch_preds.detach().cpu())
            preds = torch.cat(preds, dim=1)
            t_pred_top10 = preds.topk(10).indices
            t_pred_top10 = t_pred_top10.detach().cpu()
            top_10s.append(t_pred_top10)

            if t_filter_mask is not None:
                filter_masks.append(torch.from_numpy(t_filter_mask))

            if use_full_preds:
                full_preds.append(preds.detach())

            if isinstance(dataset, KGValidationDataset):
                t_correct_index = torch.tensor(t_correct_index)
                t_corrects.append(t_correct_index)
            if num_batches and num_batches == (i + 1):
                break

            if i % 100 == 0:
                dist.barrier(group=world)

    t_pred_top10 = torch.cat(top_10s, dim=0)
    aggregated_top10_preds = gather_results(t_pred_top10.to(local_rank), global_rank, local_rank, gather_sizes, world).detach().cpu()

    aggregated_correct_indices = None
    if isinstance(dataset, KGValidationDataset):
        t_correct_index = torch.cat(t_corrects, dim=0)
        aggregated_correct_indices = gather_results(t_correct_index.to(local_rank), global_rank, local_rank, gather_sizes, world).detach().cpu()

    aggregated_full_preds = None
    if use_full_preds:
        full_preds = torch.cat(full_preds, dim=0)
        aggregated_full_preds = gather_results(full_preds.to(local_rank), global_rank, local_rank, gather_sizes, world).detach().cpu()

    aggregated_filter_masks = None
    if filter_masks:
        filter_masks = torch.cat(filter_masks, dim=0)
        aggregated_filter_masks = gather_results(filter_masks.to(local_rank), global_rank, local_rank, gather_sizes, world).detach().cpu()

    return aggregated_top10_preds, aggregated_correct_indices, aggregated_full_preds, aggregated_filter_masks


def validate(valid_dataset: KGValidationDataset, valid_dataloader: DataLoader, model, global_rank: int,
             local_rank: int, gather_sizes: list, num_batches: int = None, world=None):
    evaluator = WikiKG90MEvaluator()
    use_full_preds = FLAGS.dataset != "wikikg90m_kddcup2021"
    top10_preds, correct_indices, full_preds, filter_mask = run_inference(valid_dataset, valid_dataloader, model,
                                                                          global_rank, local_rank, gather_sizes,
                                                                          num_batches, world,
                                                                          use_full_preds=use_full_preds)

    if global_rank == 0:
        if FLAGS.validation_attribution and FLAGS.validation_only:
            input_dict = {'h,r->t': {'t_pred_top10': top10_preds.cpu().numpy(),
                                     't_correct_index': correct_indices.cpu().numpy(),
                                     'hr': valid_dataset.hr,
                                     't_candidate': valid_dataset.t_candidate}}

            stats_dict = {}
            stats_dict_thresholds = {}
            indegrees, indegrees_thresholds = convert_stats_to_percentiles(valid_dataset.ds.indegrees)
            stats_dict['t_indegree'] = indegrees
            stats_dict_thresholds['t_indegree'] = indegrees_thresholds
            t_outdegree, t_outdegree_thresholds = convert_stats_to_percentiles(valid_dataset.ds.outdegrees)
            stats_dict['t_outdegree'] = t_outdegree
            stats_dict_thresholds['t_outdegree'] = t_outdegree_thresholds
            t_degree, t_degree_thresholds = convert_stats_to_percentiles(valid_dataset.ds.degrees)
            stats_dict['t_degree'] = t_degree
            stats_dict_thresholds['t_degree'] = t_degree_thresholds
            h_indegree, h_indegree_thresholds = convert_stats_to_percentiles(valid_dataset.ds.indegrees)
            stats_dict['h_indegree'] = h_indegree
            stats_dict_thresholds['h_indegree'] = h_indegree_thresholds
            h_outdegree, h_outdegree_thresholds = convert_stats_to_percentiles(valid_dataset.ds.outdegrees)
            stats_dict['h_outdegree'] = h_outdegree
            stats_dict_thresholds['h_outdegree'] = h_outdegree_thresholds,
            h_degree, h_degree_thresholds = convert_stats_to_percentiles(valid_dataset.ds.degrees)
            stats_dict['h_degree'] = h_degree
            stats_dict_thresholds['h_degree'] = h_degree_thresholds
            r_frequency, r_frequency_thresholds = convert_stats_to_percentiles(np.unique(valid_dataset.ds.train_r, return_counts=True)[1])
            stats_dict['r_frequency'] = r_frequency
            stats_dict_thresholds['r_frequency'] = r_frequency_thresholds

            attr_evaluator = AttributedEvaluator()
            results = attr_evaluator.eval(input_dict, stats_dict)
            pickle.dump(results, open("validation_analysis.pkl", "wb"))
            pickle.dump(stats_dict_thresholds, open("analysis_thresholds.pkl", "wb"))

        if use_full_preds:
            result_dict = compute_eval_stats(full_preds.detach().cpu().numpy(),
                                             correct_indices.detach().cpu().numpy(),
                                             filter_mask=filter_mask.detach().cpu().numpy())
            return result_dict
        else:
            input_dict = {'h,r->t': {'t_pred_top10': top10_preds.cpu().numpy(), 't_correct_index': correct_indices.cpu().numpy()}}
            result_dict = evaluator.eval(input_dict)
            return result_dict
    else:
        return None


def test(test_dataset: KGTestDataset, test_dataloader: DataLoader, model, global_rank: int,
         local_rank: int, gather_sizes: list, num_batches: int = None, world=None):
    evaluator = WikiKG90MEvaluator()
    top10_preds, _, _, _ = run_inference(test_dataset, test_dataloader, model, global_rank, local_rank, gather_sizes,
                                         num_batches, world)

    if global_rank == 0:
        print('Saving...')
        assert len(top10_preds) == len(test_dataset), f"Number of predictions is {len(top10_preds)}. Size of dataset is {len(test_dataset)}"
        input_dict = {'h,r->t': {'t_pred_top10': top10_preds}}
        evaluator.save_test_submission(input_dict=input_dict, dir_path=FLAGS.test_save_dir)
        print(f'Results saved under {FLAGS.test_save_dir}')


def full_evaluation(eval_dataset: KGEvaluationDataset, eval_dataloader: DataLoader, model, global_rank: int,
                    local_rank: int, gather_sizes: list, num_batches: int = None, world=None):
    _, correct_indices, full_preds, filter_mask = run_inference(eval_dataset, eval_dataloader, model, global_rank,
                                                                local_rank, gather_sizes, num_batches, world,
                                                                use_full_preds=True)

    if global_rank == 0:
        results = compute_eval_stats(full_preds, correct_indices, filter_mask=filter_mask)
        print(results)


def gather_results(data: torch.Tensor, global_rank, local_rank, gather_sizes, world):
    gather_list = []

    for size in gather_sizes:
        gather_list.append(torch.empty(size, *data.shape[1:], dtype=data.dtype, device=local_rank))

    dist.barrier()

    if global_rank == 0:
        gather_list[0] = data

        for p in range(1, world.size()):
            dist.recv(gather_list[p], src=p, group=world)
    else:
        assert data.shape == gather_list[global_rank].shape, "Gather size does not match data being sent. Check code for bug."
        dist.send(data, dst=0, group=world)

    return torch.cat(gather_list, dim=0)


def main(argv):
    if FLAGS.distributed:
        grank = int(os.environ['RANK'])
        ws = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        dist.init_process_group(backend=dist.Backend.NCCL,
                                init_method="tcp://{}:{}".format(master_addr, master_port), rank=grank, world_size=ws)

        setproctitle.setproctitle("KGCompletionTrainer:{}".format(grank))
        world = dist.group.WORLD

        if FLAGS.validation_only or FLAGS.test_only:
            inference_only(grank, FLAGS.local_rank, world)
        else:
            train(grank, FLAGS.local_rank, world)
        dist.destroy_process_group()
    else:
        train_single(FLAGS.device)


if __name__ == "__main__":
    app.run(main)
