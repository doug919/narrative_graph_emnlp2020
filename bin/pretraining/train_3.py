import os
import csv
import sys
import time
import json
import h5py
import logging
import argparse
import random

import dgl
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm, trange
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

from nglib.common import utils
from nglib.narrative.dataset import NarrativeGraphDataset
from nglib.narrative.dataset import batch_nid2rows
from nglib.narrative.dataset import sample_truncated_ng
from nglib.narrative.narrative_graph import get_pp_ridx2distr
from nglib.narrative.narrative_graph import get_pp_ridx2distr_coref
from nglib.models.bert_rgcn import RGCNLinkPredict, BertEventTransE, BertEventComp


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='train multi-relational models')
    parser.add_argument('ng_config', metavar='NG_CONFIG',
                        help='NG configuration')
    parser.add_argument('model_config', metavar='MODEL_CONFIG',
                        help='model configuration')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')
    parser.add_argument('--train_dir', default=None, type=str,
                        help='train files')
    parser.add_argument('--dev_dir', default=None, type=str,
                        help='dev files')
    parser.add_argument('--test_dir', default=None, type=str,
                        help='test files')

    parser.add_argument("--n_train_instances", default=-1, type=int,
                        help="number of training instances (save loading time).")

    parser.add_argument("--n_truncated_ng", default=4, type=int,
                        help="number of truncated graphs for each NG.")
    parser.add_argument("--edge_sample_rate", default=0.05, type=float,
                        help="proportion of missing edges for each NG.")
    parser.add_argument("--n_neg_per_pos", default=20, type=int,
                        help="number of negative samles per pos edge.")
    parser.add_argument('--sample_entity_only', action='store_true', default=False,
                        help='do not sample relation type')
    parser.add_argument('--sample_coref_only', action='store_true', default=False,
                        help='sample coref rel only')

    parser.add_argument("--lr", default=2e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.95, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup steps.")
    parser.add_argument("--warmup_portion", default=0.0, type=float,
                        help="Linear warmup over warmup portion.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_epochs', default=3, type=int,
                        help='number of training epochs')
    parser.add_argument("--logging_steps", type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--train_batch_size', default=4, type=int,
                        help='training batch size')
    parser.add_argument('--eval_batch_size', default=8, type=int,
                        help='evaluating batch size')

    parser.add_argument("--seed", type=int, default=135,
                        help="random seed for initialization")
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id')

    parser.add_argument('--multi_gpus', action='store_true', default=False,
                        help='use multiple GPUs for training')
    parser.add_argument('--master_addr', type=str, default="localhost",
                        help='master address for distributed training')
    parser.add_argument('--master_port', type=int, default=19191,
                        help='master port for distributed training')
    parser.add_argument('--model_name', type=str, default="ng",
                        choices=['ng', 'bert_transe', 'bert_comp'],
                        help='ng, bert_transe, bert_comp')

    parser.add_argument('--dev_coref', action='store_true', default=False,
                        help='dev model based on coref performance')
    parser.add_argument('--no_first_eval', action='store_true', default=False,
                        help='no first_evaluation (train only)')
    parser.add_argument('--no_eval', action='store_true', default=False,
                        help='no evaluation (train only)')
    parser.add_argument('--from_checkpoint', default=None, type=str,
                        help='checkpoint directory')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def set_seed(gpu, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu != -1:
        torch.cuda.manual_seed_all(seed)


def get_concat_dataset(data_dir, has_target_edges=False):
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')])
    datasets = [NarrativeGraphDataset(f, has_target_edges) for f in files]
    return ConcatDataset(datasets)


def get_batch_relations(target_edges):
    rels = [te[:, 1, :].flatten() for te in target_edges]
    rels = torch.cat(rels, dim=0)
    return rels.cpu()


def class_measure(class_cm, n_examples, rels, y_pred, y, n_classes):
    for i in range(n_classes):
        idxs = (rels == i).nonzero().flatten()
        pred = y_pred[idxs]
        gold = y[idxs]
        cm = confusion_matrix(gold, pred)
        class_cm[i] += cm

        n_pos = gold.sum()
        n_examples[i][1] += n_pos
        n_examples[i][0] += (gold.shape[0] - n_pos)


def evaluate(local_rank, model, dataloader, gpu, model_name, get_prec_recall_f1=False, logger=None, dev_ridx=-1):
    if model_name == 'bert_transe':
        out = transe_evaluate(local_rank, model, dataloader, gpu,
                             get_prec_recall_f1=get_prec_recall_f1, logger=logger, dev_ridx=dev_ridx)
    elif model_name == 'bert_comp':
        out = coref_evaluate(local_rank, model, dataloader, gpu,
                             get_prec_recall_f1=get_prec_recall_f1, logger=logger)
    else:
        out = basic_evaluate(local_rank, model, dataloader, gpu,
                             get_prec_recall_f1=get_prec_recall_f1, logger=logger, dev_ridx=dev_ridx)
    return out


def _eval_thr(thr, all_pred_scores, all_rels, all_ys, n_output_rels, dev_ridx=-1, logger=None):
    class_cm = np.zeros((n_output_rels, 2, 2), dtype=np.int64)
    n_examples = np.zeros((n_output_rels, 2), dtype=np.int64)

    all_preds = (all_pred_scores > thr).long()
    class_measure(class_cm, n_examples, all_rels, all_preds, all_ys, n_output_rels)

    # macro-averaged
    precisions, recalls = [], []
    for i in range(n_output_rels):
        tn, fp, fn, tp = class_cm[i].ravel()
        c_prec = tp / (tp + fp) if tp + fp != 0 else 0.0
        c_recall = tp / (tp + fn) if tp + fn != 0 else 0.0
        c_f1 = (2.0 * c_prec * c_recall) / (c_prec + c_recall) if c_prec + c_recall != 0 else 0.0

        precisions.append(c_prec)
        recalls.append(c_recall)
        if logger:
            logger.info('class={}, #pos={}, #neg={}, prec={}, recall={}, f1={}'.format(
                i, n_examples[i][1], n_examples[i][0], c_prec, c_recall, c_f1))
    if dev_ridx == -1:
        prec_macro = sum(precisions) / len(precisions)
        recall_macro = sum(recalls) / len(recalls)
        f1_macro = (2 * prec_macro * recall_macro) / (prec_macro + recall_macro) if \
            (prec_macro + recall_macro) != 0 else 0.0
        return prec_macro, recall_macro, f1_macro
    else:
        prec = precisions[dev_ridx]
        recall = recalls[dev_ridx]
        f1 = (2 * prec * recall) / (prec + recall) if prec + recall != 0 else 0.0
        return prec, recall, f1


def transe_evaluate(local_rank, model, dataloader, gpu, get_prec_recall_f1=False, logger=None, dev_ridx=-1):
    model.eval()
    n_output_rels = model.module.num_output_rels if \
        isinstance(model, DistributedDataParallel) else model.num_output_rels
    class_cm = np.zeros((n_output_rels, 2, 2), dtype=np.int64)
    n_examples = np.zeros((n_output_rels, 2), dtype=np.int64)
    all_rels, all_pred_scores, all_ys = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='evaluating',
                          disable=(local_rank not in [-1, 0])):
            rels = get_batch_relations(batch['target_edges'])

            # to GPU
            if gpu != -1:
                batch = NarrativeGraphDataset.to_gpu(batch, gpu)

            # forward pass
            pred_scores, y = model(mode='predict', **batch)

            all_pred_scores.append(pred_scores.cpu())
            all_ys.append(y.cpu())
            all_rels.append(rels)

    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_ys = torch.cat(all_ys, dim=0)
    all_rels = torch.cat(all_rels, dim=0)

    # find the threshold that makes the classification
    avg_thr = all_pred_scores.mean()  # the mean should a fast choice
    # search a threshold around the mean
    thr_candidates = [avg_thr + step * 0.01 * avg_thr for step in range(-20, 21)]

    n_output_rel = model.module.num_output_rels if isinstance(model, DistributedDataParallel) else model.num_output_rels
    best_thr = None
    best_metric = float('-inf')
    for thr in thr_candidates:
        prec_macro, recall_macro, f1_macro = _eval_thr(
            thr, all_pred_scores, all_rels, all_ys,
            n_output_rels, dev_ridx=dev_ridx, logger=None)
        if f1_macro > best_metric:
            best_metric = f1_macro
            best_thr = thr

    logger.info('BEST_THRESHOLD = {}'.format(best_thr))
    prec_macro, recall_macro, f1_macro = _eval_thr(
        best_thr, all_pred_scores, all_rels, all_ys,
        n_output_rels, dev_ridx=dev_ridx, logger=logger)
    if get_prec_recall_f1:
        return prec_macro, recall_macro, f1_marco
    return f1_macro


def coref_evaluate(local_rank, model, dataloader, gpu, get_prec_recall_f1=False, logger=None):
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)
    n_examples = np.zeros((2, ), dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='evaluating',
                          disable=(local_rank not in [-1, 0])):
            # to GPU
            if gpu != -1:
                batch = NarrativeGraphDataset.to_gpu(batch, gpu)

            # forward pass
            pred_scores, y = model(mode='predict', **batch)
            y_pred = (pred_scores.cpu() >= 0.5).long()
            y = y.cpu()

            # record
            cm += confusion_matrix(y, y_pred)

            n_pos = y.sum()
            n_examples[1] += n_pos
            n_examples[0] += (y.shape[0] - n_pos)

    # F1 on positive class
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if tp + fp != 0 else 0.0
    recall = tp / (tp + fn) if tp + fn != 0 else 0.0
    f1 = (2.0 * prec * recall) / (prec + recall) if prec + recall != 0 else 0.0
    logger.info('tp={}, tn={}, fp={}, fn={}'.format(tp, tn, fp, fn))
    logger.info('#pos={}, #neg={}, prec={}, recall={}, f1={}'.format(
        n_examples[1], n_examples[0], prec, recall, f1))

    if get_prec_recall_f1:
        return prec, recall, f1
    return f1


def basic_evaluate(local_rank, model, dataloader, gpu, get_prec_recall_f1=False, logger=None, dev_ridx=-1):
    model.eval()
    n_output_rels = model.module.num_output_rels if \
        isinstance(model, DistributedDataParallel) else model.num_output_rels
    class_cm = np.zeros((n_output_rels, 2, 2), dtype=np.int64)
    n_examples = np.zeros((n_output_rels, 2), dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='evaluating',
                          disable=(local_rank not in [-1, 0])):
            rels = get_batch_relations(batch['target_edges'])

            # to GPU
            if gpu != -1:
                batch = NarrativeGraphDataset.to_gpu(batch, gpu)

            # forward pass
            pred_scores, y = model(mode='predict', **batch)
            y_pred = (pred_scores.cpu() >= 0.5).long()
            y = y.cpu()
            class_measure(class_cm, n_examples, rels, y_pred, y, n_output_rels)

    # macro-averaged
    precisions, recalls = [], []
    for i in range(n_output_rels):
        tn, fp, fn, tp = class_cm[i].ravel()
        c_prec = tp / (tp + fp) if tp + fp != 0 else 0.0
        c_recall = tp / (tp + fn) if tp + fn != 0 else 0.0
        c_f1 = (2.0 * c_prec * c_recall) / (c_prec + c_recall) if c_prec + c_recall != 0 else 0.0

        precisions.append(c_prec)
        recalls.append(c_recall)
        if logger:
            logger.info('class={}, #pos={}, #neg={}, prec={}, recall={}, f1={}'.format(
                i, n_examples[i][1], n_examples[i][0], c_prec, c_recall, c_f1))

    if dev_ridx == -1:
        prec_macro = sum(precisions) / len(precisions)
        recall_macro = sum(recalls) / len(recalls)
        f1_macro = (2 * prec_macro * recall_macro) / (prec_macro + recall_macro) if \
            (prec_macro + recall_macro) != 0 else 0.0
        if get_prec_recall_f1:
            return prec_macro, recall_macro, f1_macro
        return f1_macro
    else:
        prec = precisions[dev_ridx]
        recall = recalls[dev_ridx]
        f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) != 0 else 0.0
        if get_prec_recall_f1:
            return prec, recall, f1
        return f1


def get_n_instances(dir_path):
    dataset = get_concat_dataset(dir_path)
    return len(dataset)


def get_optimizer(args, model, logger):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    if args.from_checkpoint:
        target_dir = utils.get_target_model_dir(args.from_checkpoint)
        fpath = os.path.join(target_dir, 'optimizer.pt')
        if os.path.isfile(fpath):
            logger.info('loading optimizer from {}...'.format(fpath))
            optimizer.load_state_dict(torch.load(fpath,
                                                 map_location='cpu'))
    return optimizer


def get_init_model(args, logger):
    logger.info('model_name = {}'.format(args.model_name))
    model_config = json.load(open(args.model_config))
    if args.from_checkpoint:
        target_dir = utils.get_target_model_dir(args.from_checkpoint)
        logger.info('loading model from {}...'.format(target_dir))
        if args.model_name == 'ng':
            model = RGCNLinkPredict.from_pretrained(
                target_dir,
                model_config)
        elif args.model_name == 'bert_transe':
            n_negs = args.n_neg_per_pos
            model_config['n_negs'] = n_negs
            model = BertEventTransE.from_pretrained(
                target_dir,
                model_config)
        elif args.model_name == 'bert_comp':
            n_negs = args.n_neg_per_pos
            model_config['n_negs'] = n_negs
            model = BertEventComp.from_pretrained(
                target_dir,
                model_config)
        else:
            raise NotImplementedError
    else:
        if args.model_name == 'ng':
            model = RGCNLinkPredict(**model_config)
        elif args.model_name == 'bert_transe':
            n_negs = args.n_neg_per_pos
            model_config['n_negs'] = n_negs
            model = BertEventTransE(**model_config)
        elif args.model_name == 'bert_comp':
            n_negs = args.n_neg_per_pos
            model_config['n_negs'] = n_negs
            model = BertEventComp(**model_config)
        else:
            raise NotImplementedError
    if args.gpu_id != -1:
        model.cuda(args.gpu_id)
    if args.n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.gpu_id],
                                        find_unused_parameters=True)
    return model


def get_scheduler(args, n_instances, optimizer, logger):
    t_total = (n_instances // (args.train_batch_size * max(1, args.n_gpus))
               // args.gradient_accumulation_steps) * args.n_epochs
    warmup_steps = 0
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    elif args.warmup_portion > 0:
        warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", warmup_steps)
    if args.from_checkpoint:
        target_dir = utils.get_target_model_dir(args.from_checkpoint)
        fpath = os.path.join(target_dir, 'scheduler.pt')
        if os.path.isfile(fpath):
            logger.info('loading scheduler from {}...'.format(fpath))
            scheduler.load_state_dict(torch.load(fpath,
                                                 map_location='cpu'))
    return scheduler


def train_sample_truncated_ng(ng_edges,
                        input_ids, input_masks, token_type_ids, target_idxs,
                        nid2rows,
                        n_instances,
                        ent_pred_ridxs,
                        pred_pred_ridxs,
                        ep_rtype_rev,
                        n_truncated_ng,
                        edge_sample_rate,
                        n_neg_per_pos,
                        pp_ridx2distr,
                        coref_ridx,
                        sample_entity_only):
    all_target_edges = []
    all_truncated_gs = []
    for i in range(len(ng_edges)):
        n_nodes = nid2rows[i].shape[0]
        target_edges, input_edges = \
            sample_truncated_ng(ng_edges[i],
                                n_nodes,
                                ent_pred_ridxs,
                                pred_pred_ridxs,
                                ep_rtype_rev,
                                n_truncated_ng,
                                edge_sample_rate,
                                n_neg_per_pos,
                                pp_ridx2distr,
                                coref_ridx,
                                sample_entity_only)
        all_target_edges.append(target_edges)

        truncated_gs = []
        for j in range(len(input_edges)):
            new_edges = input_edges[j]

            g = dgl.DGLGraph()
            g.add_nodes(n_nodes)
            g.add_edges(new_edges[0].long(), new_edges[2].long())

            edge_types = new_edges[1].long()

            # TODO: avoid storing norms
            # calculate norms, instead of using the pre-calculated
            for k in range(new_edges.shape[1]):
                nid2 = new_edges[2][k]
                new_edges[3][k] = 1.0 / g.in_degree(nid2)
            edge_norms = new_edges[3].unsqueeze(1)
            g.edata.update({'rel_type': edge_types})
            g.edata.update({'norm': edge_norms})
            truncated_gs.append(g)
        all_truncated_gs.append(truncated_gs)

    batch = {
        'bg': all_truncated_gs,
        'input_ids': input_ids,
        'input_masks': input_masks,
        'token_type_ids': token_type_ids,
        'target_idxs': target_idxs,
        'nid2rows': nid2rows,
        'target_edges': all_target_edges,
        'n_instances': n_instances
    }
    return batch


def train_collate(samples):
    ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows = \
        map(list, zip(*samples))
    if len(ng_edges) <= 1:
        input_ids = bert_inputs[0][0]
        input_masks = bert_inputs[0][1]
        token_type_ids = bert_inputs[0][2]
        tidxs = bert_target_idxs[0]

        n_instances = [input_ids.shape[0]]

        nid2rows = bert_nid2rows[0]
    else:
        input_ids = torch.cat([bi[0] for bi in bert_inputs], dim=0)
        input_masks = torch.cat([bi[1] for bi in bert_inputs], dim=0)
        token_type_ids = torch.cat([bi[2] for bi in bert_inputs], dim=0)
        tidxs = torch.cat(bert_target_idxs, dim=0)

        n_instances = [bi[0].shape[0] for bi in bert_inputs]

    nid2rows = bert_nid2rows # list

    batch = {
        'ng_edges': ng_edges, # will be used to sample target edges and bg
        'input_ids': input_ids,
        'input_masks': input_masks,
        'token_type_ids': token_type_ids,
        'target_idxs': tidxs,
        'nid2rows': nid2rows,
        'n_instances': n_instances
    }
    return batch


def test_collate(samples):
    ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows, target_edges, input_edges, gn = \
        map(list, zip(*samples))
    if len(ng_edges) <= 1:
        # batch size should be one for saving memory
        # ng_edges = ng_edges[0]

        input_ids = bert_inputs[0][0]
        input_masks = bert_inputs[0][1]
        token_type_ids = bert_inputs[0][2]
        tidxs = bert_target_idxs[0]

        n_instances = [input_ids.shape[0]]
    else:
        # ng_edges = torch.cat(ng_edges, dim=0)

        # batch bert inputs in one dim, and keep n_instances per graph
        input_ids = torch.cat([bi[0] for bi in bert_inputs], dim=0)
        input_masks = torch.cat([bi[1] for bi in bert_inputs], dim=0)
        token_type_ids = torch.cat([bi[2] for bi in bert_inputs], dim=0)
        tidxs = torch.cat(bert_target_idxs, dim=0)

        n_instances = [bi[0].shape[0] for bi in bert_inputs]

    # target_edges = torch.stack(target_edges, dim=0)

    nid2rows = bert_nid2rows # list

    all_truncated_gs = []
    for i in range(len(input_edges)):
        ies = input_edges[i]
        n2r = nid2rows[i]
        n_nodes = n2r.shape[0]

        truncated_gs = []
        for new_edges in ies:
            g = dgl.DGLGraph()
            g.add_nodes(n_nodes)
            g.add_edges(new_edges[0].long(), new_edges[2].long())

            edge_types = new_edges[1].long()

            # TODO: avoid storing norms
            # calculate norms, instead of using the pre-calculated
            for k in range(new_edges.shape[1]):
                nid2 = new_edges[2][k]
                new_edges[3][k] = 1.0 / g.in_degree(nid2)
            edge_norms = new_edges[3].unsqueeze(1)
            g.edata.update({'rel_type': edge_types})
            g.edata.update({'norm': edge_norms})
            truncated_gs.append(g)
        all_truncated_gs.append(truncated_gs)

    batch = {
        # 'ng_edges': ng_edges,
        'bg': all_truncated_gs,
        'input_ids': input_ids,
        'input_masks': input_masks,
        'token_type_ids': token_type_ids,
        'target_idxs': tidxs,
        'nid2rows': nid2rows,
        'target_edges': target_edges,
        'n_instances': n_instances
    }
    return batch


def prepare_train_dataset(f, local_rank, args):
    train_dataset = NarrativeGraphDataset(f)
    if args.n_gpus > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.n_gpus,
            rank=local_rank
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size, # fix 1 for sampling
            shuffle=False,
            num_workers=1, # 1 is safe for hdf5
            collate_fn=train_collate,
            sampler=train_sampler)
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size, # fix 1 for sampling
            shuffle=True,
            num_workers=1, # 1 is safe for hdf5
            collate_fn=train_collate)
        train_sampler = None
    return train_dataset, train_sampler, train_dataloader


def save_model(model, optimizer, scheduler, output_dir, step, logger):
    dir_path = 'best_model_{}'.format(step)
    model_to_save = model.module \
        if isinstance(model, DistributedDataParallel) \
        else model
    model_to_save.save_pretrained(output_dir, dir_path)
    dir_path = os.path.join(output_dir, dir_path)
    logger.info('save model {}...'.format(dir_path))
    torch.save(optimizer.state_dict(),
               os.path.join(dir_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(),
               os.path.join(dir_path, "scheduler.pt"))


def train(rank, args):
    logger = utils.get_root_logger(args, log_fname='log_rank{}'.format(rank))
    if args.n_gpus > 1:
        local_rank = rank
        args.gpu_id = rank
    else:
        local_rank = -1

    if args.n_gpus > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.n_gpus,
            rank=local_rank
        )

    set_seed(args.gpu_id, args.seed) # in distributed training, this has to be same for all processes

    logger.info('local_rank = {}, n_gpus = {}'.format(local_rank, args.n_gpus))
    logger.info('n_epochs = {}'.format(args.n_epochs))

    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # initialize training essentials
    if local_rank in [-1, 0]:
        dev_dataset = get_concat_dataset(args.dev_dir, has_target_edges=True)
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=test_collate,
            pin_memory=False
        )

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        tb_writer = SummaryWriter(comment='_{}'.format(args.output_dir))

    train_files = [os.path.join(args.train_dir, f)
                   for f in os.listdir(args.train_dir) if f.endswith('.h5')]
    t1 = time.time()
    if args.n_train_instances != -1: # save loading time
        n_instances = args.n_train_instances
    else:
        n_instances = get_n_instances(args.train_dir)
    logger.info('get_n_instances = {}: {}s'.format(n_instances, time.time()-t1))


    # NG config
    ng_config = json.load(open(args.ng_config))
    assert ng_config["config_target"] == "narrative_graph"
    rtype2idx = ng_config['rtype2idx']
    if ng_config['no_entity']:
        ep_rtype_rev = {}
        ent_pred_ridxs = set()
    else:
        ep_rtype_rev = {rtype2idx[v]: rtype2idx[k] for k, v in
                              ng_config['entity_predicate_rtypes'].items()}
        ent_pred_ridxs = set(ep_rtype_rev.keys())
    coref_ridx = rtype2idx['cnext']

    n_rtypes = len(rtype2idx)
    pred_pred_ridxs = set(range(n_rtypes)) - ent_pred_ridxs
    if args.sample_coref_only:
        pp_ridx2distr = get_pp_ridx2distr_coref(ng_config)
    else:
        pp_ridx2distr = get_pp_ridx2distr(ng_config)

    # model config
    model_config = json.load(open(args.model_config, 'r'))

    model = get_init_model(args, logger)
    optimizer = get_optimizer(args, model, logger)
    scheduler = get_scheduler(args, n_instances, optimizer, logger)

    # training
    dev_ridx = coref_ridx if args.dev_coref else -1
    if local_rank in [-1, 0]:
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", args.n_epochs)
        logger.info("  Training batch size = %d", args.train_batch_size)
        logger.info("  Evaluation batch size = %d", args.eval_batch_size)
        logger.info("  Accu. train batch size = %d",
                        args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Weight Decay = {}".format(args.weight_decay))
        logger.info("  Learning Rate = {}".format(args.lr))

        # first eval result
        if args.no_first_eval:
            best_metric = 0.0
        else:
            best_metric = evaluate(
                local_rank, model, dev_dataloader, args.gpu_id,
                model_name=args.model_name, logger=logger, dev_ridx=dev_ridx)
        logger.info('start dev_metric = {}'.format(best_metric))
    else:
        best_metric = 0.0

    step = 0
    prev_acc_loss, acc_loss = 0.0, 0.0
    t1 =  time.time()
    model.zero_grad()
    for i_epoch in range(args.n_epochs):
        logger.info('========== Epoch {} =========='.format(i_epoch))

        t2 = time.time()
        random.shuffle(train_files) # shuffle files
        for i_file, f in enumerate(train_files):
            logger.debug('file = {}'.format(f))
            logger.info('{} / {} files completed'.format(i_file, len(train_files)))

            # load one dataset (file) in memory
            t3 = time.time()
            train_dataset, train_sampler, train_dataloader = \
                prepare_train_dataset(f, local_rank, args)

            # training on batches
            if args.n_gpus > 1:
                train_sampler.set_epoch(i_epoch)
            for train_batch in tqdm(train_dataloader, desc='training on one file',
                                    disable=True):
                model.train()
                # truncate graph
                train_batch = train_sample_truncated_ng(
                    **train_batch,
                    ent_pred_ridxs=ent_pred_ridxs,
                    pred_pred_ridxs=pred_pred_ridxs,
                    ep_rtype_rev=ep_rtype_rev,
                    n_truncated_ng=args.n_truncated_ng,
                    edge_sample_rate=args.edge_sample_rate,
                    n_neg_per_pos=args.n_neg_per_pos,
                    pp_ridx2distr=pp_ridx2distr,
                    coref_ridx=coref_ridx,
                    sample_entity_only=args.sample_entity_only
                )

                # to GPU
                if args.gpu_id != -1:
                    train_batch = NarrativeGraphDataset.to_gpu(train_batch, args.gpu_id)

                # forward pass:
                loss = model(mode='loss', **train_batch)

                if args.n_gpus > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # backward pass
                loss.backward()

                acc_loss += loss.item()

                # accumation
                if (step + 1) % args.gradient_accumulation_steps == 0: # ignore the last accumulation
                    # update params
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    step += 1

                    # loss
                    if local_rank in [-1, 0]:
                        if args.logging_steps > 0 and step % args.logging_steps == 0:
                            cur_loss = (acc_loss - prev_acc_loss) / args.logging_steps
                            logger.info('train_loss={}, step={}, time={}s'.format(cur_loss,
                                                                                  step,
                                                                                  time.time()-t1))
                            tb_writer.add_scalar('train_loss', cur_loss, step)
                            tb_writer.add_scalar('lr', scheduler.get_last_lr()[0])

                            # evaluate
                            if not args.no_eval:
                                dev_metric = evaluate(
                                    local_rank, model, dev_dataloader, args.gpu_id,
                                    model_name=args.model_name, logger=logger, dev_ridx=dev_ridx)
                                logger.info('dev_metric={}'.format(dev_metric))
                                if best_metric < dev_metric:
                                    best_metric = dev_metric

                                    # save
                                    save_model(model, optimizer, scheduler, args.output_dir, step, logger)
                            else:
                                # simply save model
                                save_model(model, optimizer, scheduler, args.output_dir, step, logger)
                            prev_acc_loss = acc_loss

            logger.info('done file: {} s'.format(time.time() - t3))
        logger.info('done epoch: {} s'.format(time.time() - t2))

    logger.info('done training: {} s'.format(time.time() - t1))

    t1 = time.time()
    if local_rank in [-1, 0]:
        tb_writer.close()

        del model, dev_dataloader

        # test
        if args.test_dir is not None:
            test_metric = test(local_rank, args.output_dir, args.test_dir, logger, args)
            logger.info('test_metric = {}'.format(test_metric))
    logger.info('done testing: {} s'.format(time.time() - t1))


def test(local_rank, model_dir, data_dir, logger, args):
    # search models
    target_model_dir = utils.get_target_model_dir(model_dir)

    # load model
    logger.info('loading test model {}...'.format(target_model_dir))
    model_config = json.load(open(args.model_config, 'r'))
    model = RGCNLinkPredict.from_pretrained(target_model_dir, model_config)
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)

    # load data
    logger.info('loading test dataset {}...'.format(data_dir))
    test_dataset = get_concat_dataset(data_dir, has_target_edges=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=test_collate,
        pin_memory=False
    )

    # test
    test_metric = evaluate(local_rank, model, test_dataloader, args.gpu_id,
                           model_name=args.model_name, get_prec_recall_f1=True, logger=logger)
    return test_metric


def main():
    if args.multi_gpus: # use all GPUs in parallel
        assert torch.cuda.is_available()
        args.n_gpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
    elif args.gpu_id != -1:
        args.n_gpus = 1
    else:
        args.n_gpus = 0

    assert args.train_dir is not None, 'must provide train_dir'
    if not args.no_eval:
        assert args.dev_dir is not None, 'must provide dev_dir'

    # start training
    if args.n_gpus > 1:
        mp.spawn(train, nprocs=args.n_gpus, args=(args,))
    else:
        train(args.gpu_id, args)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    main()
