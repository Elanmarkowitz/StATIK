import json
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
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from ogb.lsc import WikiKG90MEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, SubsetRandomSampler
from tqdm import tqdm

from attributed_eval import AttributedEvaluator, convert_stats_to_percentiles
from data.data_classes import get_testing_dataset, get_training_dataset, get_validation_dataset, KGInferenceDataset, \
    KGBaseDataset, get_training_inference_dataset
from data.data_loading import TrainingCollateFunction, InferenceCollateQueryFunction, InferenceCollateTargetFunction
from data.data_processing import load_processed_data, KGProcessedDataset
from evaluation import compute_eval_stats
from model.kg_completion_gnn import KGCompletionGNN, ModelConfig
from utils import load_model, load_opt_checkpoint, save_checkpoint, save_model

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "wikikg90m_kddcup2021", "Specifies dataset from [wikikg90m_kddcup2021, wordnet-mlj12]")
flags.DEFINE_string("root_data_dir", "/nas/home/elanmark/data", "Root data dir for installing the ogb dataset")

flags.DEFINE_integer("num_workers", 0, "Number of workers for the dataloader.")
flags.DEFINE_integer("local_rank", 0, "Local rank, used for torch distributed launch.")
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
flags.DEFINE_integer("validate_every", 1, "How many iterations to do between each single batch validation.")
flags.DEFINE_integer("validation_batches", None, "Max number of batches to do for each validation check (default: None).")
flags.DEFINE_integer("valid_batch_size", 1, "Batch size for validation (does all t_candidates at once).")
flags.DEFINE_integer("epochs", 1, "Num epochs to train for")
flags.DEFINE_float("margin", 1.0, "Margin in transE loss")
flags.DEFINE_string("decoder", "MLP+TransE", "Choose decoder from [TBA]")
flags.DEFINE_string("language_model", "bert-base-cased", "Name of language model to use for encoding [from huggingface]")
flags.DEFINE_string("encoder", "ours_parallel", "Encoder type to use [ours_parallel, ours_sequential, BLP, StAR]")
flags.DEFINE_float("dropout", 0.0, "Dropout on input features.")
flags.DEFINE_float("warmup", 0.2, "Warmup period as proportion of total training")

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
    train_collate_fn = TrainingCollateFunction(train_dataset, max_neighbors=FLAGS.samples_per_node,
                                               num_negatives=FLAGS.neg_samples, encoder_method=FLAGS.encoder)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers, sampler=train_sampler,
                              collate_fn=train_collate_fn)

    target_dataset = get_validation_dataset(dataset).target_mode()
    target_subset = partition_dataset_by_rank(target_dataset, global_rank, world)
    target_collate_fn = InferenceCollateTargetFunction(target_dataset, FLAGS.samples_per_node, FLAGS.encoder)
    target_dataloader = DataLoader(target_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                   collate_fn=target_collate_fn)

    eval_queries = np.random.choice(np.arange(0, dataset.num_entities), (10000,), replace=False)
    train_eval_dataset = get_training_inference_dataset(dataset).query_mode()
    train_eval_collate_fn_tail = InferenceCollateQueryFunction(train_eval_dataset, FLAGS.samples_per_node,
                                                               head_prediction=False, encoder_method=FLAGS.encoder)
    train_eval_collate_fn_head = InferenceCollateQueryFunction(train_eval_dataset, FLAGS.samples_per_node,
                                                               head_prediction=False, encoder_method=FLAGS.encoder)
    train_eval_dataset = Subset(train_eval_dataset, eval_queries)  # don't use all queries for mrr calculation
    train_eval_subset = partition_dataset_by_rank(train_eval_dataset, global_rank, world)
    train_target_dataset = get_training_inference_dataset(dataset).target_mode()
    train_target_subset = partition_dataset_by_rank(train_target_dataset, global_rank, world)
    train_target_collate_fn = InferenceCollateTargetFunction(train_target_dataset, FLAGS.samples_per_node, FLAGS.encoder)
    train_target_dataloader = DataLoader(train_target_subset, batch_size=FLAGS.valid_batch_size,
                                         num_workers=FLAGS.num_workers, collate_fn=train_target_collate_fn)
    train_eval_dataloader_tail = DataLoader(train_eval_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                            collate_fn=train_eval_collate_fn_tail)
    train_eval_dataloader_head = DataLoader(train_eval_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                            collate_fn=train_eval_collate_fn_head)

    valid_dataset = get_validation_dataset(dataset).query_mode()
    valid_subset = partition_dataset_by_rank(valid_dataset, global_rank, world)
    valid_collate_fn_tail = InferenceCollateQueryFunction(valid_dataset, FLAGS.samples_per_node, head_prediction=False,
                                                          encoder_method=FLAGS.encoder)
    valid_dataloader_tail = DataLoader(valid_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                       collate_fn=valid_collate_fn_tail)
    valid_collate_fn_head = InferenceCollateQueryFunction(valid_dataset, FLAGS.samples_per_node, head_prediction=True,
                                                          encoder_method=FLAGS.encoder)
    valid_dataloader_head = DataLoader(valid_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                       collate_fn=valid_collate_fn_head)
    if FLAGS.checkpoint is not None:
        model = load_model(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint), ignore_state_dict=(global_rank != 0))
    else:
        model = KGCompletionGNN(dataset.relation_feat, dataset.num_relations, dataset.feature_dim, ModelConfig(FLAGS))

    model.to(local_rank)

    # saved = torch.load('checkpoints/wn18rr_transe_star_best_model.pkl', map_location="cpu")
    # model.load_state_dict(saved['state_dict'], strict=False)

    ddp_model = DDP(model, device_ids=[local_rank], process_group=world, find_unused_parameters=True)
    loss_fn = model.get_loss_fn(margin=FLAGS.margin)
    opt = optim.AdamW(ddp_model.parameters(), lr=FLAGS.lr, betas=(.9, .98))

    fake_opt = optim.AdamW(ddp_model.parameters(), lr=FLAGS.lr)
    scheduler = get_linear_schedule_with_warmup(opt,
                                                num_warmup_steps=int(FLAGS.warmup * len(train_loader) * FLAGS.epochs),
                                                num_training_steps=len(train_loader) * FLAGS.epochs)

    start_epoch = 0
    if FLAGS.checkpoint:
        start_epoch = load_opt_checkpoint(os.path.join(CHECKPOINT_DIR, FLAGS.checkpoint), opt, scheduler)

    moving_average_loss = torch.tensor(1.0, device=local_rank)
    max_mrr = 0
    max_token_length = 0
    for epoch in range(start_epoch, FLAGS.epochs):
        if global_rank == 0:
            print(f'Epoch {epoch}')
        for i, (batch, queries, positive_targets, negative_targets, neg_filter) in enumerate(tqdm(train_loader)):
            ddp_model.train()
            batch = prepare_batch_for_model(batch, dataset)
            batch = move_batch_to_device(batch, local_rank)
            input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
            # print(f'max prev token size {max_token_length}, curr token length {input_ids.shape[1]}')
            pos_scores, neg_scores = ddp_model(input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query,
                                               entity_feat, query_nodes, r_relatives, is_head_prediction,
                                               queries=queries, positive_targets=positive_targets,
                                               negative_targets=negative_targets)

            loss = loss_fn(pos_scores, neg_scores, neg_filter=neg_filter)

            # max_token_length = max(input_ids.shape[1], max_token_length)

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
                          f"Moving Avg Loss={moving_average_loss.cpu().numpy():.5f}, "
                          f"Current Loss={loss.detach().cpu().numpy():.5f}")

        if (epoch + 1) % FLAGS.validate_every == 0:
            if FLAGS.dataset != "Wikidata5M":
                train_t_res, train_h_res = perform_model_validation(
                    ddp_model, train_eval_dataset, train_target_dataset, train_eval_dataloader_tail,
                    train_eval_dataloader_head, train_target_dataloader, global_rank, local_rank, world)
                if global_rank == 0:
                    mrr_tail = train_t_res['mrr_filtered']
                    mrr_head = train_h_res['mrr_filtered']
                    mrr = 0.5 * mrr_tail + 0.5 * mrr_head
                    print(f'Train MRR = {mrr}, t_pred MRR = {mrr_tail}, h_pred MRR = {mrr_head}')

            valid_t_res, valid_h_res = perform_model_validation(
                ddp_model, valid_dataset, target_dataset, valid_dataloader_tail, valid_dataloader_head,
                target_dataloader, global_rank, local_rank, world)
            if global_rank == 0:
                mrr_tail = valid_t_res['mrr_filtered']
                mrr_head = valid_h_res['mrr_filtered']
                mrr = 0.5 * mrr_tail + 0.5 * mrr_head
                if mrr > max_mrr:
                    max_mrr = mrr
                    save_model(ddp_model.module, os.path.join(CHECKPOINT_DIR, f'{FLAGS.name}_best_model.pkl'))
                print(f'Validation MRR = {mrr}, t_pred MRR = {mrr_tail}, h_pred MRR = {mrr_head}, best MRR = {max_mrr}')

        if global_rank == 0:
            save_checkpoint(ddp_model.module, epoch + 1, opt, scheduler, os.path.join(CHECKPOINT_DIR, f"{FLAGS.name}_e{epoch}.pkl"))


def perform_model_validation(model, eval_dataset, target_dataset, dataloader_tail, dataloader_head, target_dataloader,
                             global_rank, local_rank, world):
    model.eval()
    eval_gather_sizes = calculate_gather_sizes(eval_dataset, world, FLAGS.validation_batches, FLAGS.valid_batch_size)
    target_gather_sizes = calculate_gather_sizes(target_dataset, world)
    if isinstance(eval_dataset, Subset):
        eval_dataset = eval_dataset.dataset
    t_res = validate(eval_dataset, dataloader_tail, target_dataloader, model, global_rank,
                     local_rank, eval_gather_sizes, target_gather_sizes, num_batches=FLAGS.validation_batches,
                     world=world)
    h_res = validate(eval_dataset, dataloader_head, target_dataloader, model, global_rank,
                     local_rank, eval_gather_sizes, target_gather_sizes, num_batches=FLAGS.validation_batches,
                     world=world)
    return t_res, h_res


def partition_dataset_by_rank(dataset, global_rank, world):
    num_ranks = world.size()
    idxs_per_rank = math.ceil(len(dataset) / num_ranks)
    start_idx = global_rank * idxs_per_rank
    end_idx = (global_rank + 1) * idxs_per_rank if ((global_rank + 1) * idxs_per_rank <= len(dataset)) else len(dataset)
    rank_idxs = torch.arange(start_idx, end_idx, dtype=torch.long).tolist()
    # print(f"Global rank {global_rank} processing dataset from {rank_idxs[0]} through {rank_idxs[-1]}")

    subset = Subset(dataset, rank_idxs)
    return subset


def calculate_gather_sizes(dataset, world, num_batches=None, batch_size=None):
    num_ranks = world.size()
    if num_batches is None or num_batches < 0:
        gather_sizes = [math.ceil(len(dataset) / num_ranks)] * (num_ranks - 1)
        last_size = len(dataset) - sum(gather_sizes)
        gather_sizes.append(last_size)
    else:
        assert len(dataset) >= (num_batches * batch_size * num_ranks), "Too many validation batches for validation dataset size."
        gather_sizes = [batch_size * num_batches] * num_ranks

    return gather_sizes


def inference_only(global_rank, local_rank, world):
    assert FLAGS.test_only or FLAGS.validation_only, "Must run validation set or test set if running inference."
    assert not (FLAGS.test_only and FLAGS.validation_only), "Can only run on one of test or validation when doing inference."
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    base_dataset = load_processed_data(FLAGS.root_data_dir, FLAGS.dataset)
    if FLAGS.validation_only:
        eval_dataset = get_validation_dataset(base_dataset).query_mode()
        target_dataset = get_validation_dataset(base_dataset).target_mode()
    else:
        eval_dataset = get_testing_dataset(base_dataset).query_mode()
        target_dataset = get_testing_dataset(base_dataset).target_mode()

    target_subset = partition_dataset_by_rank(target_dataset, global_rank, world)
    target_collate_fn = InferenceCollateTargetFunction(target_dataset, FLAGS.samples_per_node, FLAGS.encoder)
    target_dataloader = DataLoader(target_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                   collate_fn=target_collate_fn)
    target_gather_sizes = calculate_gather_sizes(target_dataset, world)

    eval_subset = partition_dataset_by_rank(eval_dataset, global_rank, world)
    eval_collate_fn_head = InferenceCollateQueryFunction(eval_dataset, FLAGS.samples_per_node, True, FLAGS.encoder)
    eval_dataloader_head = DataLoader(eval_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                 collate_fn=eval_collate_fn_head)
    eval_collate_fn_tail = InferenceCollateQueryFunction(eval_dataset, FLAGS.samples_per_node, False, FLAGS.encoder)
    eval_dataloader_tail = DataLoader(eval_subset, batch_size=FLAGS.valid_batch_size, num_workers=FLAGS.num_workers,
                                 collate_fn=eval_collate_fn_tail)
    eval_gather_sizes = calculate_gather_sizes(eval_dataset, world, FLAGS.validation_batches, FLAGS.valid_batch_size)

    assert FLAGS.model_path is not None, 'Must be supplied with model to do inference.'
    model = load_model(FLAGS.model_path, ignore_state_dict=(global_rank != 0))

    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], process_group=world)
    ddp_model.eval()

    result = {}
    for mode, loader in zip(["head", "tail"], [eval_dataloader_head, eval_dataloader_tail]):
        if FLAGS.validation_only:
            result[mode] = validate(eval_dataset, loader, target_dataloader, ddp_model, global_rank, local_rank,
                                    eval_gather_sizes, target_gather_sizes, FLAGS.validation_batches, world)
        else:
            result[mode] = test(eval_dataset, loader, target_dataloader, ddp_model, global_rank, local_rank,
                                eval_gather_sizes, target_gather_sizes, FLAGS.validation_batches, world)
    if global_rank == 0:
        result["mean"] = {}
        for key in result["head"].keys():
            result["mean"][key] = 0.5 * result["head"][key] + 0.5 * result["tail"][key]
        print(json.dumps(result, indent=2, sort_keys=False))


def run_inference(dataset: KGInferenceDataset, dataloader: DataLoader, target_loader: DataLoader, model,
                  global_rank: int, local_rank: int, eval_gather_sizes: list, target_gather_sizes: list,
                  num_batches: int = None, world=None, use_full_preds=False):
    model.eval()

    with torch.no_grad():
        model.module.encode_only(True)
        target_embeds = []
        for i, batch in enumerate(target_loader):
            batch = prepare_batch_for_model(batch, dataset.base_ds)
            batch = move_batch_to_device(batch, local_rank)
            input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
            batch_target_embeds = model(input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query,
                                        entity_feat, query_nodes, r_relatives, is_head_prediction)
            target_embeds.append(batch_target_embeds.cpu())
        target_embeds = torch.cat(target_embeds, dim=0)

        full_target_embeds = gather_results(target_embeds, global_rank, local_rank, target_gather_sizes, world, all_gather=True)

    full_pos_scores = []
    full_target_scores = []
    full_filter_masks = []
    with torch.no_grad():
        model.module.encode_only(False)
        for i, (batch, true_target, target_filter_mask) in enumerate(dataloader):
            batch = prepare_batch_for_model(batch, dataset.base_ds)
            batch = move_batch_to_device(batch, local_rank)
            input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query, entity_set, entity_feat, query_nodes, r_relatives, is_head_prediction = batch
            pos_scores, neg_scores = model(input_ids, token_type_ids, attention_mask, ht_tensor, r_tensor, r_query,
                                           entity_feat, query_nodes, r_relatives, is_head_prediction,
                                           queries=torch.logical_not(true_target), target_embeddings=full_target_embeds)
            full_pos_scores.append(pos_scores.cpu())
            full_target_scores.append(neg_scores.cpu())
            full_filter_masks.append(target_filter_mask)

            if num_batches and num_batches == (i + 1):
                break

            if i % 100 == 0:
                dist.barrier(group=world)

        full_pos_scores = torch.cat(full_pos_scores, dim=0)
        agg_pos_scores = gather_results(full_pos_scores, global_rank, local_rank, eval_gather_sizes, world).cpu()

        full_target_scores = torch.cat(full_target_scores, dim=0)
        agg_target_scores = gather_results(full_target_scores, global_rank, local_rank, eval_gather_sizes, world).cpu()

        full_filter_masks = torch.cat(full_filter_masks, dim=0)
        agg_filter_masks = gather_results(full_filter_masks, global_rank, local_rank, eval_gather_sizes, world).cpu()

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

    result_dict = None
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
    return result_dict


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
            result_dict = compute_eval_stats(pos_scores.detach().cpu().numpy(),
                                             target_scores.detach().cpu().numpy(),
                                             filter_mask=filter_mask.detach().cpu().numpy())
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

    device = local_rank if dist.get_backend(world) == dist.Backend.NCCL else 'cpu'

    for size in gather_sizes:
        gather_list.append(torch.empty(size, *data.shape[1:], dtype=data.dtype, device=device))

    dist.barrier()

    if global_rank == 0:
        gather_list[0] = data.to(device)

        for p in range(1, world.size()):
            dist.recv(gather_list[p], src=p, group=world)
    else:
        assert data.shape == gather_list[global_rank].shape, \
            f"Gather size {gather_list[global_rank].shape} does not match data being sent {data.shape}. Check code for bug."
        dist.send(data.to(device), dst=0, group=world)

    dist.barrier()
    result = torch.cat(gather_list, dim=0)
    if all_gather:
        dist.broadcast(result, 0)
    dist.barrier()
    return result


def main(argv):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    grank = int(os.environ['RANK'])
    ws = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    backend = dist.Backend.GLOO if FLAGS.validation_only or FLAGS.test_only else dist.Backend.NCCL
    dist.init_process_group(backend=backend,
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
