import os
import csv
import sys
import time
import json
import h5py
import pickle as pkl
import logging
import argparse
import random
from copy import deepcopy
from collections import OrderedDict
from itertools import permutations

import dgl
import torch
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
from stanza.server import CoreNLPClient

from nglib.common import utils
from nglib.narrative.dataset import NarrativeGraphDataset
from nglib.models.bert_rgcn import RGCNLinkPredict, BERTNarrativeGraph
from nglib.models.bert_rgcn import BertEventComp, BertEventTransE
from nglib.narrative.narrative_graph import get_ng_bert_tokenizer


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='test intrinsic evaluations')
    parser.add_argument('task', metavar='TASK',
                        choices=[
                            'pp_coref_next',
                            'pp_discourse_link',
                            'triplet'
                        ],
                        help='task to run')
    parser.add_argument('ng_config', metavar='NG_CONFIG',
                        help='ng config file')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input dir')
    parser.add_argument('question_dir', metavar='QUESTION_DIR',
                        help='question dir')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')

    parser.add_argument('--model_name', type=str,
                        choices=[
                            'ng', 'bert', 'bert_comp', 'bert_transe'],
                        default='ng',
                        help='model class name')
    parser.add_argument('--model_config', type=str, default=None,
                        help='model configuration')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='model dir')

    parser.add_argument("--bert_weight_name",
                        default='google/bert_uncased_L-2_H-128_A-2', type=str,
                        help="bert weight version")
    parser.add_argument('--is_cased', action='store_true', default=False,
                        help='BERT is case sensitive')

    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id')
    parser.add_argument('--test_n_graphs', type=int, default=-1,
                        help='test n graphs')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


class BERTLinkPredict(torch.nn.Module):
    '''naive BERT baseline for comparison
    '''
    def __init__(self, in_dim, h_dim, num_narrative_rels, num_output_rels,
                 bert_weight_name,
                 num_hidden_layers=2, dropout=0.2, reg_param=0,
                 use_gate=False, class_weights=[]):
        super(BERTLinkPredict, self).__init__()
        self.bert_ng = BERTNarrativeGraph(bert_weight_name)
        self.cossim_f = torch.nn.CosineSimilarity()

    def forward(self, bg, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows, target_edges, mode='embeddings', n_instances=None,
                **kwargs):
        assert mode == 'predict'
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)

        # split by nodes
        assert isinstance(nid2rows, list)
        # list of nid2rows
        node_embs = []
        for i in range(len(nid2rows)):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])
            node_embs.append(ne)

        # no RGCN, so edge doesn't matter. we simply copy k times
        n_truncated_gs = len(bg[0])
        node_embs = node_embs[0].unsqueeze(0).expand(n_truncated_gs, -1, -1)
        all_target_edges = []
        for i in range(len(target_edges)):
            for j in range(n_truncated_gs):
                all_target_edges.append(target_edges[i][j])
        out = self.predict(
            node_embs,
            all_target_edges)
        return out

    def cosine_sim(self, embedding, target_edges):
        h = embedding[target_edges[0]]
        t = embedding[target_edges[2]]
        score = self.cossim_f(h, t)
        return score

    def predict(self, embeddings, target_edges):
        y_pred, y = [], []
        for emb, te in zip(embeddings, target_edges):
            labels = te[3]
            y.append(labels)

            score = self.cosine_sim(emb, te)
            y_pred.append(score)
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)
        return y_pred, y


def load_model(model_name, model_dir, model_config, gpu_id=-1):
    # search models
    target_model_dir = utils.get_target_model_dir(model_dir)

    # load model
    model_config = json.load(open(model_config, 'r'))
    if model_name ==  'ng':
        logger.info('loading test model {}...'.format(target_model_dir))
        model = RGCNLinkPredict.from_pretrained(target_model_dir, model_config)
    elif model_name == 'bert':
        model = BERTLinkPredict(**model_config)
    elif model_name == 'bert_transe':
        logger.info('loading test model {}...'.format(target_model_dir))
        model = BertEventTransE.from_pretrained(target_model_dir, model_config)
    elif model_name == 'bert_comp':
        logger.info('loading test model {}...'.format(target_model_dir))
        model = BertEventComp.from_pretrained(target_model_dir, model_config)
    else:
        raise ValueError('not implemented model_name: {}'.format(model_name))

    if gpu_id != -1:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)
    return model


def get_dgl_graph_list(input_edges, n_nodes):
    gs = []
    for edges in input_edges:
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.add_edges(edges[0].long(), edges[2].long())

        edge_types = edges[1].long()
        for i in range(edges.shape[1]):
            nid2 = int(edges[2][i])
            edges[3][i] = 1.0 / g.in_degree(nid2)
        edge_norms = edges[3].unsqueeze(1)

        g.edata.update({'rel_type': edge_types})
        g.edata.update({'norm': edge_norms})
        gs.append(g)
    return gs


def get_ng_inputs(test_data):
    ng_edges = torch.from_numpy(
        test_data['ng_edges'].astype('float32'))
    binputs = torch.from_numpy(
        test_data['bert_inputs'].astype('int64'))
    target_idxs = torch.from_numpy(
        test_data['bert_target_idxs'].astype('int64'))
    nid2rows = torch.from_numpy(
        test_data['bert_nid2rows'].astype('int64'))

    target_edges = torch.from_numpy(
        test_data['target_edges'].astype('int64'))
    input_edges = torch.from_numpy(
        test_data['input_edges'].astype('float32'))
    return ng_edges, binputs, target_idxs, nid2rows, target_edges, input_edges


def _task_pp_discourse_link_v2(model, test_data, results, **kwargs):
    if kwargs['q'] is None:
        return
    input_edges, correct, choices = kwargs['q']
    _, g_binputs, g_target_idxs, g_nid2rows, _, _ = \
        get_ng_inputs(test_data)
    input_edges = torch.from_numpy(input_edges.astype('float32'))
    disc_ridx_list = kwargs['disc_ridx_list']
    gold_target_edge = choices[correct]

    # create target edge choices
    correct_idx = None
    target_edges = torch.ones((4, len(disc_ridx_list)), dtype=torch.int64)
    target_edges[0, :] = gold_target_edge[0]
    target_edges[2, :] = gold_target_edge[2]
    for i, ridx in enumerate(disc_ridx_list):
        if ridx == gold_target_edge[1]:
            correct_idx = i
        target_edges[1, i] = ridx
    target_edges = target_edges.unsqueeze(0)

    n_nodes = g_nid2rows.shape[0]
    gs = get_dgl_graph_list([input_edges], n_nodes)

    batch = {
        'bg': [gs],
        'input_ids': g_binputs[0],
        'input_masks': g_binputs[1],
        'token_type_ids': g_binputs[2],
        'target_idxs': g_target_idxs,
        'nid2rows': [g_nid2rows],
        'n_instances': [g_binputs.shape[1]],
        'target_edges': [target_edges]
    }
    if 'predict' not in results:
        results['predict'] = []
    if 'y' not in results:
        results['y'] = []
    if 'scores' not in results:
        results['scores'] = []
    with torch.no_grad():
        if args.gpu_id !=  -1:
            batch = NarrativeGraphDataset.to_gpu(batch, args.gpu_id)
        pred_scores, y = model(mode='predict', **batch)
        pred_scores, y = pred_scores.cpu(), y.cpu()

        pred_idx = torch.argmax(pred_scores).item()

        logger.debug('scores={}, pred={}, y={}'.format(pred_scores, pred_idx, correct_idx))

        # note that we put the real rtype idx here
        results['predict'].append(disc_ridx_list[pred_idx])
        results['y'].append(disc_ridx_list[correct_idx])
        results['scores'].append(pred_scores.tolist())


def _task_pp_any_next_v2(model, test_data, results, **kwargs):
    if kwargs['q'] is None:
        return
    input_edges, correct, choices = kwargs['q']
    _, g_binputs, g_target_idxs, g_nid2rows, _, _ = \
        get_ng_inputs(test_data)
    input_edges = torch.from_numpy(input_edges.astype('float32'))

    n_choices = len(choices)
    target_edges = torch.ones((4, n_choices), dtype=torch.int64)
    for i in range(n_choices):
        for j in range(3):
            target_edges[j, i] = choices[i][j]
    target_edges = target_edges.unsqueeze(0)

    n_nodes = g_nid2rows.shape[0]
    gs = get_dgl_graph_list([input_edges], n_nodes)

    batch = {
        'bg': [gs],
        'input_ids': g_binputs[0],
        'input_masks': g_binputs[1],
        'token_type_ids': g_binputs[2],
        'target_idxs': g_target_idxs,
        'nid2rows': [g_nid2rows],
        'n_instances': [g_binputs.shape[1]],
        'target_edges': [target_edges]
    }
    with torch.no_grad():
        if args.gpu_id !=  -1:
            batch = NarrativeGraphDataset.to_gpu(batch, args.gpu_id)
        pred_scores, y = model(mode='predict', **batch)
        pred_scores, y = pred_scores.cpu(), y.cpu()

        pred = torch.argmax(pred_scores)
        if 'predict' not in results:
            results['predict'] = []
        if 'y' not in results:
            results['y'] = []

        logger.debug('scores={}, pred={}, y={}'.format(pred_scores, pred, correct))
        results['predict'].append(pred.item())
        results['y'].append(correct)


def _task_triplet_classification(model, test_data, results, **kwargs):
    # predict for one graph
    count_correct = 0

    # because we have to get target edges from pkl for some tasks
    # we don't use the NarrativeGraphDataset here
    # instead, we re-write it for each task
    _, binputs, target_idxs, nid2rows, target_edges, input_edges = \
        get_ng_inputs(test_data)
    n_nodes = nid2rows.shape[0]
    gs = get_dgl_graph_list(input_edges, n_nodes)

    gold_rtypes = []
    for te in target_edges:
        gold_rtypes.append(te[1])
    gold_rtypes = torch.cat(gold_rtypes, dim=0).tolist()

    with torch.no_grad():
        batch = {
            'bg': [gs],
            'input_ids': binputs[0],
            'input_masks': binputs[1],
            'token_type_ids': binputs[2],
            'target_idxs': target_idxs,
            'nid2rows': [nid2rows],
            'n_instances': [binputs.shape[1]],
            'target_edges': [target_edges]
        }
        if args.gpu_id !=  -1:
            batch = NarrativeGraphDataset.to_gpu(batch, args.gpu_id)
        pred_scores, y = model(mode='predict', **batch)
        pred_scores, y = pred_scores.cpu(), y.cpu()

        thr = 0.5
        y_pred = (pred_scores >= thr).long().tolist()
        y = y.tolist()
        for r, yp, gold in zip(gold_rtypes, y_pred, y):
            if r not in results:
                results[r] = []
            results[r].append((yp, gold))


def mrr_recall_k_score(y, scores, disc_ridx_list, k):
    tp = 0
    mrr = 0.0
    ridx2idx = {r: i for i, r in enumerate(disc_ridx_list)}
    for ridx, s in zip(y, scores):
        # rs = [1/x for x in s]
        rs = []
        for x in s:
            if x != 0:
                rs.append(1/x)
            else:
                rs.append(float('inf'))

        rank = rankdata(rs)
        target = ridx2idx[ridx]
        r = rank[target]
        mrr += 1 / r

        if r <= k:
            tp += 1

    mrr = mrr / len(y)
    recall_k = float(tp) / len(y)
    return mrr, recall_k


def eval_task(ng_config, model, task_func, q_key=None):
    rtype2idx = ng_config['rtype2idx']
    idx2rtype = {v: k for k, v in rtype2idx.items()}
    if ng_config['no_entity']:
        ep_rtype_rev = {}
        ent_pred_ridxs = set()
    else:
        ep_rtype_rev = {rtype2idx[v]: rtype2idx[k] for k, v in
                              ng_config['entity_predicate_rtypes'].items()}
        ent_pred_ridxs = set(ep_rtype_rev.keys())
    n_rtypes = len(rtype2idx)
    pred_pred_ridxs = set(range(n_rtypes)) - ent_pred_ridxs
    disc_pred_pred_ridxs = pred_pred_ridxs - {rtype2idx['next'], rtype2idx['cnext']}
    disc_ridx_list = sorted(list(disc_pred_pred_ridxs)) # 8-13

    t1 = time.time()
    results = {}
    count_gids = 0
    fs = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.h5')])
    for f in fs:
        fpath = os.path.join(args.input_dir, f)
        logger.info('processing {}...'.format(fpath))

        qf = '.'.join(f.split('.')[:-1])
        q_fpath = os.path.join(args.question_dir, 'q_{}.pkl'.format(qf))
        logger.info('loading {}...'.format(q_fpath))
        questions = pkl.load(open(q_fpath, 'rb'))

        t2 = time.time()
        fr = h5py.File(fpath, 'r')
        for gn in tqdm(fr.keys()):
            q = questions[gn][q_key] if q_key else None
            gid = int(gn.split('_')[-1])

            # load graph data
            bert_inputs = fr[gn]['bert_inputs'][:]
            bert_target_idxs = fr[gn]['bert_target_idxs'][:]
            bert_nid2rows = fr[gn]['bert_nid2rows'][:]
            ng_edges = fr[gn]['ng_edges'][:]
            input_edges = fr[gn]['input_edges'][:]
            target_edges = fr[gn]['target_edges'][:]
            test_data = {
                'ng_edges': ng_edges,
                'bert_inputs': bert_inputs,
                'bert_target_idxs': bert_target_idxs,
                'bert_nid2rows': bert_nid2rows,
                'input_edges': input_edges,
                'target_edges': target_edges
            }

            # update results
            task_func(model, test_data, results,
                      q=q,
                      rtype2idx=rtype2idx,
                      disc_ridx_list=disc_ridx_list)
            count_gids += 1
            if args.test_n_graphs != -1 and count_gids >= args.test_n_graphs:
                break
        fr.close()
        logger.info('file done: {} s'.format(time.time()-t2))
        if args.test_n_graphs != -1 and count_gids >= args.test_n_graphs:
            break

    logger.info('task done: {} s'.format(time.time()-t1))
    logger.info('#graphs = {}'.format(count_gids))


    # output
    if args.task == 'triplet':
        all_precisions, all_recalls = [], []
        all_n_pos, all_n_neg = [], []
        fpath = os.path.join(args.output_dir, 'report.csv')
        with open(fpath, 'w') as fw:
            writer = csv.writer(fw, delimiter=',', quotechar='"')
            writer.writerow(['', 'precision', 'recall', 'f1', 'n_pos', 'n_neg'])
            for ridx, res in results.items():
                yp = [p[0] for p in res]
                y = [p[1] for p in res]
                logger.info('\nRTYPE = {}'.format(idx2rtype[ridx]))
                report = classification_report(y, yp, digits=6, output_dict=True)
                logger.info(report)

                if '1' in report:
                    prec, recall, f1 = report['1']['precision'], report['1']['recall'], report['1']['f1-score']
                    n_pos, n_neg = report['1']['support'], report['0']['support']
                    writer.writerow([idx2rtype[ridx], prec*100, recall*100, f1*100, n_pos, n_neg])

                    all_precisions.append(prec)
                    all_recalls.append(recall)
                    all_n_pos.append(n_pos)
                    all_n_neg.append(n_neg)

        # macro-averaged
        prec_macro = sum(all_precisions) / len(all_precisions)
        recall_macro = sum(all_recalls) / len(all_recalls)
        f1_macro = (2 * prec_macro * recall_macro) / (prec_macro + recall_macro) if \
            (prec_macro + recall_macro) != 0 else 0.0
        all_n_pos = sum(all_n_pos)
        all_n_neg = sum(all_n_neg)
        logger.info('macro: #pos={}, #neg={}, prec={}, recall={}, f1={}'.format(
            all_n_pos, all_n_neg, prec_macro, recall_macro, f1_macro))

    elif args.task in ['pp_coref_next']:
        acc = accuracy_score(results['y'], results['predict'])
        logger.info('#instances={}, accuracy={}'.format(len(results['y']), acc))
    elif args.task == 'pp_discourse_link':
        logger.info(classification_report(results['y'], results['predict'], digits=6, labels=disc_ridx_list))
        k = 3
        mrr, recall_k = mrr_recall_k_score(results['y'], results['scores'], disc_ridx_list, k=k)
        prfs = precision_recall_fscore_support(results['y'], results['predict'], average='micro')
        logger.info('micro PRFS = {}'.format(prfs))
        logger.info('MRR = {}'.format(mrr))
        logger.info('recall@{} = {}'.format(k, recall_k))


def main():
    config = json.load(open(args.ng_config))
    assert config["config_target"] == "narrative_graph"

    model = load_model(
        args.model_name, args.model_dir, args.model_config, args.gpu_id)
    model.eval()

    if args.task == 'triplet':
        # PP_DISCOURSE_TRIPLET task
        eval_task(config, model, _task_triplet_classification)
    elif args.task == 'pp_coref_next':
        # sample PP_COREF_NEXT task
        eval_task(config, model, _task_pp_any_next_v2, 'pp_coref_next')
    elif args.task == 'pp_discourse_link':
        # sample PP_DISCOURSE_LINK_TYPE task
        eval_task(config, model, _task_pp_discourse_link_v2, 'pp_discourse_next')
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
