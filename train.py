import math
import random
from typing import Union

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
from data.data_classes import get_testing_dataset, get_training_dataset, get_validation_dataset, KGInferenceDataset, \
    KGBaseDataset
from data.data_loading import TrainingCollateFunction, InferenceCollateQueryFunction, InferenceCollateTargetFunction
from data.data_processing import load_processed_data, KGProcessedDataset
from evaluation import compute_eval_stats
from model.kg_completion_gnn import KGCompletionGNN
from utils import load_model, load_opt_checkpoint, save_checkpoint, save_model

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "wikikg90m_kddcup2021", "Specifies dataset from [wikikg90m_kddcup2021, wordnet-mlj12]")
flags.DEFINE_string("root_data_dir", "/nas/home/elanmark/data", "Root data dir for installing the ogb dataset")

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
flags.DEFINE_float("dropout", 0.0, "Dropout on input features.")

flags.DEFINE_bool('validation_only', False, 'Whether or not to do a complete inference run across the validation set')
flags.DEFINE_bool('test_only', False, 'Whether or not to do a complete inference run across the test set')
flags.DEFINE_string('model_path', None, 'Path where the model is saved (inference only)')
flags.DEFINE_string("test_save_dir", "test_submissions", "Directory to save test results file in.")
flags.DEFINE_bool("validation_attribution", False, "Whether to perform validation attribution on full validation runs")

flags.DEFINE_bool("predict_heads", True, "Whether to predict heads during inference.")

CHECKPOINT_DIR = "checkpoints"


def prepare_batch_for_model(batch, dataset: Union[KGProcessedDataset, KGBaseDataset], save_batch=False):
    input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
    if entity_feat is None:
        entity_feat = torch.from_numpy(dataset.entity_feat[entity_set.numpy()]).float()

    batch = input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction
    if save_batch:
        pickle.dump(batch, open('sample_batch.pkl', 'wb'))
    return batch


def move_batch_to_device(batch, device, *args):
    input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    ht_tensor = ht_tensor.to(device)
    r_tensor = r_tensor.to(device)
    r_query = r_query.to(device)
    entity_feat = entity_feat.to(device)
    r_relatives = r_relatives.to(device)
    is_head_prediction = is_head_prediction.to(device)
    batch = input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction
    if args:
        args = [arg.to(device) for arg in args]
        return batch, args
    return batch


def train(global_rank, local_rank, world):
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataset = load_processed_data(FLAGS.root_data_dir, FLAGS.dataset)

    train_dataset = get_training_dataset(dataset)
    train_sampler = DistributedSampler(train_dataset, rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers, sampler=train_sampler,
                              collate_fn=TrainingCollateFunction(train_dataset, max_neighbors=FLAGS.samples_per_node,
                                                                 num_negatives=FLAGS.neg_samples))

    target_dataset = get_validation_dataset(dataset).target_mode()
    target_subset = partition_dataset_by_rank(target_dataset, global_rank, world)
    target_collate_fn = InferenceCollateTargetFunction(target_dataset, FLAGS.samples_per_node)
    target_dataloader = DataLoader(target_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                   collate_fn=target_collate_fn)

    valid_dataset = get_validation_dataset(dataset).query_mode()
    valid_sampler = DistributedSampler(valid_dataset, rank=global_rank, shuffle=False)
    valid_collate_fn_tail = InferenceCollateQueryFunction(valid_dataset, FLAGS.samples_per_node, head_prediction=False)
    valid_dataloader_tail = DataLoader(valid_dataset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                       sampler=valid_sampler, drop_last=True, collate_fn=valid_collate_fn_tail)
    valid_collate_fn_head = InferenceCollateQueryFunction(valid_dataset, FLAGS.samples_per_node, head_prediction=True)
    valid_dataloader_head = DataLoader(valid_dataset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                       sampler=valid_sampler, drop_last=True, collate_fn=valid_collate_fn_head)
    if FLAGS.checkpoint is not None:
        model = load_model(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint), ignore_state_dict=(global_rank != 0))
    else:
        model = KGCompletionGNN(dataset.relation_feat, dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim,
                                FLAGS.layers, decoder=FLAGS.decoder, dropout=FLAGS.dropout)

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
        for i, (batch, queries, positive_targets, negative_targets) in enumerate(tqdm(train_loader)):
            ddp_model.train()
            batch = prepare_batch_for_model(batch, dataset)
            batch = move_batch_to_device(batch, local_rank)
            input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
            pos_scores, neg_scores = ddp_model(input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query,
                                               entity_feat, query_nodes, r_relatives, is_head_prediction,
                                               queries=queries, positive_targets=positive_targets,
                                               negative_targets=negative_targets)

            loss = loss_fn(pos_scores, neg_scores)

            moving_average_loss = .99 * moving_average_loss + 0.01 * loss.detach()

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            if (FLAGS.print_freq > 0 and (i + 1) % FLAGS.print_freq == 0) or ((i + 1) == len(train_loader)):
                dist.all_reduce(moving_average_loss, group=world)
                moving_average_loss /= dist.get_world_size()

                if global_rank == 0:
                    print(f"Iteration={i}/{len(train_loader)}, "
                          f"Moving Avg Loss={moving_average_loss.cpu().numpy():.5f}")

        if (epoch + 1) % FLAGS.validate_every == 0:
            ddp_model.eval()
            eval_gather_sizes = calculate_gather_sizes(valid_dataset, world, FLAGS.validation_batches, FLAGS.valid_batch_size)
            target_gather_sizes = calculate_gather_sizes(target_dataset, world)
            result = validate(valid_dataset, valid_dataloader_tail, target_dataloader, ddp_model, global_rank,
                              local_rank, eval_gather_sizes, target_gather_sizes, num_batches=FLAGS.validation_batches, world=world)
            result2 = validate(valid_dataset, valid_dataloader_head, target_dataloader, ddp_model, global_rank,
                               local_rank, eval_gather_sizes, target_gather_sizes, num_batches=FLAGS.validation_batches, world=world)
            if global_rank == 0:
                mrr_tail = result['mrr']
                mrr_head = result2['mrr']
                result['mrr'] = 0.5 * result['mrr'] + 0.5 * result2['mrr']
                mrr = result['mrr']
                if mrr > max_mrr:
                    max_mrr = mrr
                    save_model(ddp_model.module, os.path.join(CHECKPOINT_DIR, f'{FLAGS.name}_best_model.pkl'))

                print('Current MRR = {}, t_pred MRR = {}, h_pred MRR = {}, Best MRR = {}'.format(mrr, mrr_tail, mrr_head, max_mrr))

        if global_rank == 0:
            save_checkpoint(ddp_model.module, epoch + 1, opt, scheduler, os.path.join(CHECKPOINT_DIR, f"{FLAGS.name}_e{epoch}.pkl"))


def partition_dataset_by_rank(dataset, global_rank, world):
    num_ranks = world.size()
    idxs_per_rank = math.ceil(len(dataset) / num_ranks)
    start_idx = global_rank * idxs_per_rank
    end_idx = (global_rank + 1) * idxs_per_rank if ((global_rank + 1) * idxs_per_rank <= len(dataset)) else len(dataset)
    rank_idxs = torch.arange(start_idx, end_idx, dtype=torch.long).tolist()
    print(f"Global rank {global_rank} processing dataset from {rank_idxs[0]} through {rank_idxs[-1]}")

    subset = Subset(dataset, rank_idxs)
    return subset


def calculate_gather_sizes(dataset, world, num_batches=None, batch_size=None):
    num_ranks = world.size()
    if num_batches is None:
        gather_sizes = [math.ceil(len(dataset) / num_ranks)] * (num_ranks - 1)
        last_size = len(dataset) - sum(gather_sizes)
        gather_sizes.append(last_size)
    else:
        gather_sizes = [batch_size * num_batches] * num_ranks

    return gather_sizes


def inference_only(global_rank, local_rank, world):
    assert FLAGS.test_only or FLAGS.validation_only, "Must run validation set or test set if running inference."
    assert not (FLAGS.test_only and FLAGS.validation_only), "Can only run on one of test or validation when doing inference."
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if FLAGS.validation_batches < 0:
        FLAGS.validation_batches = None

    base_dataset = load_processed_data(FLAGS.root_data_dir, FLAGS.dataset)
    if FLAGS.validation_only:
        eval_dataset = get_validation_dataset(base_dataset).query_mode()
        target_dataset = get_validation_dataset(base_dataset).target_mode()
    else:
        eval_dataset = get_testing_dataset(base_dataset).query_mode()
        target_dataset = get_testing_dataset(base_dataset).target_mode()

    target_subset = partition_dataset_by_rank(target_dataset, global_rank, world)
    target_collate_fn = InferenceCollateTargetFunction(target_dataset, FLAGS.samples_per_node)
    target_dataloader = DataLoader(target_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                   collate_fn=target_collate_fn)
    target_gather_sizes = calculate_gather_sizes(target_dataset, world)

    eval_subset = partition_dataset_by_rank(eval_dataset, global_rank, world)
    eval_collate_fn = InferenceCollateQueryFunction(eval_dataset, FLAGS.samples_per_node, FLAGS.predict_heads)
    eval_dataloader = DataLoader(eval_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                 collate_fn=eval_collate_fn)
    eval_gather_sizes = calculate_gather_sizes(eval_dataset, world, FLAGS.validation_batches, FLAGS.valid_batch_size)

    assert FLAGS.model_path is not None, 'Must be supplied with model to do inference.'
    model = load_model(FLAGS.model_path, ignore_state_dict=(global_rank != 0))

    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], process_group=world)
    ddp_model.eval()

    if FLAGS.validation_only:
        result = validate(eval_dataset, eval_dataloader, target_dataloader, ddp_model, global_rank, local_rank,
                          eval_gather_sizes, target_gather_sizes, FLAGS.validation_batches, world)
    else:
        result = test(eval_dataset, eval_dataloader, target_dataloader, ddp_model, global_rank, local_rank,
                      eval_gather_sizes, target_gather_sizes, FLAGS.validation_batches, world)


def run_inference(dataset: KGInferenceDataset, dataloader: DataLoader, target_loader: DataLoader, model,
                  global_rank: int, local_rank: int, eval_gather_sizes: list, target_gather_sizes: list,
                  num_batches: int = None, world=None, use_full_preds=False):
    model.eval()

    with torch.no_grad():
        model.module.encode_only(True)
        target_embeds = []
        for i, batch in enumerate(target_loader):
            batch = prepare_batch_for_model(batch, dataset.base_dataset)
            batch = move_batch_to_device(batch, local_rank)
            input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
            batch_target_embeds = model(input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query,
                                        entity_feat, query_nodes, r_relatives, is_head_prediction)
            target_embeds.append(batch_target_embeds)
        target_embeds = torch.cat(target_embeds, dim=0)

        full_target_embeds = gather_results(target_embeds, global_rank, local_rank, target_gather_sizes, world, all_gather=True)

    full_pos_scores = []
    full_target_scores = []
    full_filter_masks = []
    with torch.no_grad():
        model.module.encode_only(False)
        for i, (batch, true_target, target_filter_mask) in enumerate(dataloader):
            batch = prepare_batch_for_model(batch, dataset.base_dataset)
            batch = move_batch_to_device(batch, local_rank)
            input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
            pos_scores, neg_scores = model(input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query,
                                           entity_feat, query_nodes, r_relatives, is_head_prediction,
                                           queries=torch.logical_not(true_target), target_embeddings=full_target_embeds)
            full_pos_scores.append(pos_scores.cpu())
            full_target_scores.append(neg_scores.cpu())
            full_filter_masks.append(target_filter_mask.cpu())

            if num_batches and num_batches == (i + 1):
                break

            if i % 100 == 0:
                dist.barrier(group=world)

        full_pos_scores = torch.cat(full_pos_scores, dim=0)
        agg_pos_scores = gather_results(full_pos_scores.to(local_rank), global_rank, local_rank, eval_gather_sizes, world).cpu()

        full_target_scores = torch.cat(full_target_scores, dim=0)
        agg_target_scores = gather_results(full_target_scores.to(local_rank), global_rank, local_rank, eval_gather_sizes, world).detach().cpu()

        full_filter_masks = torch.cat(full_filter_masks, dim=0)
        agg_filter_masks = gather_results(full_filter_masks.to(local_rank), global_rank, local_rank, eval_gather_sizes, world).detach().cpu()

    return agg_pos_scores, agg_target_scores, agg_filter_masks


def validate(valid_dataset: KGInferenceDataset, valid_dataloader: DataLoader, target_dataloader: DataLoader, model,
             global_rank: int, local_rank: int, eval_gather_sizes: list, target_gather_sizes: list,
             num_batches: int = None, world=None):
    evaluator = WikiKG90MEvaluator()
    use_full_preds = FLAGS.dataset != "wikikg90m_kddcup2021"
    pos_scores, target_scores, filter_mask = run_inference(valid_dataset, valid_dataloader, target_dataloader, model,
                                                           global_rank, local_rank, eval_gather_sizes,
                                                           target_gather_sizes, num_batches, world,
                                                           use_full_preds=use_full_preds)

    if global_rank == 0:
        if FLAGS.validation_attribution and FLAGS.validation_only: # TODO: fix this for refactor
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
            # filter_mask[:, valid_dataset.test_entities] = -1  # TODO: add this to get_testing_dataset
            result_dict = compute_eval_stats(pos_scores.detach().cpu().numpy(),
                                             target_scores.detach().cpu().numpy(),
                                             filter_mask=filter_mask.detach().cpu().numpy())
        else: # TODO: fix this for refactor
            input_dict = {'h,r->t': {'t_pred_top10': top10_preds.cpu().numpy(), 't_correct_index': correct_indices.cpu().numpy()}}
            result_dict = evaluator.eval(input_dict)
        print('Validation ' + ' '.join([f'{k}={result_dict[k]}' for k in result_dict.keys()]))
        return result_dict
    else:
        return None


def test(test_dataset, test_dataloader: DataLoader, target_dataloader: DataLoader, model, global_rank: int,
         local_rank: int, eval_gather_sizes: list, target_gather_sizes: list, num_batches: int = None, world=None):
    evaluator = WikiKG90MEvaluator()
    use_full_preds = FLAGS.dataset != "wikikg90m_kddcup2021"
    pos_scores, target_scores, filter_mask = run_inference(test_dataset, test_dataloader, target_dataloader, model,
                                                           global_rank, local_rank, eval_gather_sizes,
                                                           target_gather_sizes, num_batches, world,
                                                           use_full_preds=use_full_preds)
    result_dict = None
    if global_rank == 0:
        if use_full_preds:
            filter_mask[:, test_dataset.ds.valid_entities] = -1
            result_dict = compute_eval_stats(pos_scores.detach().cpu().numpy(),
                                             target_scores.detach().cpu().numpy(),
                                             filter_mask=filter_mask.detach().cpu().numpy())
            print('Test ' + ' '.join([f'{k}={result_dict[k]}' for k in result_dict.keys()]))
        else:
            print('Saving...')
            assert len(top10_preds) == len(test_dataset), f"Number of predictions is {len(top10_preds)}. Size of dataset is {len(test_dataset)}"
            input_dict = {'h,r->t': {'t_pred_top10': top10_preds}}
            evaluator.save_test_submission(input_dict=input_dict, dir_path=FLAGS.test_save_dir)
            print(f'Results saved under {FLAGS.test_save_dir}')
            result_dict = {}
    return result_dict


def gather_results(data: torch.Tensor, global_rank, local_rank, gather_sizes, world, all_gather=False) -> torch.Tensor:
    gather_list = []

    for size in gather_sizes:
        gather_list.append(torch.empty(size, *data.shape[1:], dtype=data.dtype, device=local_rank))

    dist.barrier()

    if global_rank == 0:
        gather_list[0] = data

        for p in range(1, world.size()):
            dist.recv(gather_list[p], src=p, group=world)
    else:
        assert data.shape == gather_list[global_rank].shape, \
            f"Gather size {gather_list[global_rank].shape} does not match data being sent {data.shape}. Check code for bug."
        dist.send(data, dst=0, group=world)

    dist.barrier()
    result = torch.cat(gather_list, dim=0)
    if all_gather:
        dist.broadcast(result, 0)

    return result


def main(argv):
    # torch.multiprocessing.set_sharing_strategy('file_system')
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



if __name__ == "__main__":
    app.run(main)
