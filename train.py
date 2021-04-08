from absl import app, flags
import tqdm
import contextlib
import pickle
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from ogb.lsc import WikiKG90MEvaluator

from data.data_loading import load_dataset, WikiKG90MProcessedDataset, Wiki90MValidationDataset, Wiki90MTestDataset
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
flags.DEFINE_integer("validate_every", 10000, "How many iterations to do between each single batch validation.")

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


def train():
    DEVICE = torch.device(FLAGS.device)
    scaler = GradScaler()

    if not DEBUGGING_MODEL:
        dataset = load_dataset(FLAGS.root_data_dir)
        train_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True,
                                  num_workers=FLAGS.num_workers,
                                  collate_fn=dataset.get_collate_fn(max_neighbors=FLAGS.samples_per_node))
        model = KGCompletionGNN(dataset.num_relations, dataset.feature_dim, FLAGS.embed_dim, FLAGS.layers)
        model.to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=FLAGS.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50,
                                                         verbose=True, threshold_mode='rel', cooldown=10)
        moving_average_loss = torch.tensor(1.0)
        moving_average_acc = torch.tensor(0.5)
        moving_avg_rank = torch.tensor(10.0)

        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            model.train()
            batch = prepare_batch_for_model(batch, dataset)
            batch = move_batch_to_device(batch, DEVICE)
            ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels = batch
            with autocast() if FLAGS.device == "cuda" else contextlib.suppress():
                preds = model(ht_tensor_batch, r_tensor, entity_feat, relation_feat, queries)
                loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels.float())

            correct = torch.eq((preds > 0).long().flatten(), labels)
            score_1 = preds[labels==1].detach().cpu().flatten()[0]
            score_0 = torch.topk(preds[labels==0].detach().cpu().flatten().float()[:100], k=9).values
            rank = 1 + (score_1 < score_0).sum()
            moving_avg_rank = .9995 * moving_avg_rank + .0005 * rank.float()

            training_acc = correct.float().mean()
            moving_average_loss = .999 * moving_average_loss + 0.001 * loss.detach().cpu()
            moving_average_acc = .99 * moving_average_acc + 0.01 * training_acc.detach().cpu()
            print(f"loss={loss.detach().cpu().numpy():.5f}, avg={moving_average_loss.numpy():.5f}, "
                  f"train_acc={training_acc.detach().cpu().numpy():.3f}, avg={moving_average_acc.numpy():.3f}, "
                  f"rank={rank} "
                  f"avg_rank={moving_avg_rank}")
            opt.zero_grad()
            scaler.scale(loss).backward() if FLAGS.device == "cuda" else loss.backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step(moving_average_loss)
            if (i + 1) % FLAGS.validate_every == 0:
                validate(dataset, model, single_itr=True)

    else:  # TODO: make part of this a function call to avoid code duplication
        import pickle
        batch = pickle.load(open('sample_batch.pkl', 'rb'))
        batch = move_batch_to_device(batch, DEVICE)
        ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels = batch
        model = KGCompletionGNN(1315, 768, FLAGS.embed_dim, FLAGS.layers)  # num ent 87143637
        model.to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=FLAGS.lr)
        model.train()
        with autocast():
            preds = model(ht_tensor_batch, r_tensor, entity_feat, relation_feat, queries)
            loss = F.binary_cross_entropy_with_logits(preds.flatten(), labels.float())
            breakpoint()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        breakpoint()


def validate(dataset: WikiKG90MProcessedDataset, model: KGCompletionGNN, single_itr=False):
    DEVICE = torch.device(FLAGS.device)
    evaluator = WikiKG90MEvaluator()
    valid_dataset = Wiki90MValidationDataset(dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, num_workers=0, shuffle=True,
                                  collate_fn=valid_dataset.get_eval_collate_fn(max_neighbors=FLAGS.samples_per_node))
    model.eval()
    top_10s = []
    t_corrects = []
    with torch.no_grad():
        for i, (batch, t_correct_index) in enumerate(tqdm.tqdm(valid_dataloader)):
            batch_preds = []
            for subbatch in tqdm.tqdm(batch):
                subbatch = prepare_batch_for_model(subbatch, valid_dataset.ds)
                subbatch = move_batch_to_device(subbatch, DEVICE)
                ht_tensor, ht_tensor_batch, r_tensor, entity_set, entity_feat, relation_feat, node_id_to_batch, queries, labels = subbatch
                with autocast() if FLAGS.device == "cuda" else contextlib.suppress():
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
    train()


if __name__ == "__main__":
    app.run(main)
