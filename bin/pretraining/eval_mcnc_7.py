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
from nglib.narrative.dataset import NarrativeGraphDataset, NarrativeSequenceDataset
from nglib.narrative.narrative_graph import get_ng_bert_tokenizer
from nglib.models.bert_rgcn import RGCNLinkPredict, BERTNarrativeGraph
from nglib.models.bert_rgcn import BertEventComp, BertEventTransE, BertEventLSTM


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='test MCNC evaluation')
    parser.add_argument('ng_config', metavar='NG_CONFIG',
                        help='ng config file')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input dir')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')

    parser.add_argument('--model_name', type=str,
                        choices=[
                            'ng', 'bert', 'bert_comp', 'bert_transe', 'lstm'],
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
    parser.add_argument('--score_pooling', type=str, default='last', choices=['mean', 'last', 'max'],
                        help='scoring options')

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
    elif model_name == 'lstm':
        logger.info('loading test model {}...'.format(target_model_dir))
        model = BertEventLSTM.from_pretrained(
            target_model_dir,
            model_config)
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

    coref_nids = torch.from_numpy(
        test_data['coref_nids'].astype('int64'))
    return ng_edges, binputs, target_idxs, nid2rows, coref_nids


def eval_one_mcnc_question_lstm(model, test_data, results, q, coref_ridx, score_pooling):
    correct, choices, target_edge_idx, chain_info = q
    ng_edges, bert_inputs, target_idxs, nid2rows, coref_nids = \
        get_ng_inputs(test_data)

    coref_nids = [m['nid'] for m in chain_info]
    coref_nids = torch.LongTensor(coref_nids)

    n_choices = len(choices)
    neg_coref_nids = torch.LongTensor(
        [choices[j][2] for j in range(n_choices) if j != correct])

    batch = {
        'input_ids': bert_inputs[0],
        'input_masks': bert_inputs[1],
        'token_type_ids': bert_inputs[2],
        'target_idxs': target_idxs,
        'nid2rows': [nid2rows],
        'n_instances': [bert_inputs.shape[1]],
        'coref_nids': [coref_nids],
        'neg_coref_nids': [neg_coref_nids]
    }
    with torch.no_grad():
        if args.gpu_id !=  -1:
            batch = NarrativeSequenceDataset.to_gpu(batch, args.gpu_id)
        pred_scores, y = model(mode='predict', **batch)

        # the model will always put the correct choice at index 0
        pred = torch.argmax(pred_scores).cpu()
        y_gold = torch.argmax(y).cpu()

        if 'predict' not in results:
            results['predict'] = []
        if 'y' not in results:
            results['y'] = []

        logger.debug('scores={}, pred={}, y={}'.format(pred_scores, pred, y_gold))
        results['predict'].append(pred.item())
        results['y'].append(y_gold.item())


def eval_one_mcnc_question(model, test_data, results, q, coref_ridx, score_pooling):
    correct, choices, target_edge_idx, chain_info = q
    ng_edges, bert_inputs, target_idxs, nid2rows, coref_nids = \
        get_ng_inputs(test_data)

    # prepare input edges
    input_edges = torch.cat(
        (ng_edges[:, :target_edge_idx], ng_edges[:, target_edge_idx+1:]),
        dim=1
    )
    n_nodes = nid2rows.shape[0]
    gs = get_dgl_graph_list([input_edges], n_nodes)

    # prepare target edges
    chain_len = coref_nids.shape[0]
    n_choices = len(choices)

    if score_pooling == 'last':
        d = n_choices
    else:
        d = (chain_len-1) * n_choices

    target_edges = torch.ones((4, d), dtype=torch.int64)
    target_edges[1, :] = coref_ridx
    if score_pooling != 'last':
        for i in range(n_choices):
            ch = choices[i]
            for j in range(chain_len-1):
                head = coref_nids[j]

                k = i * (chain_len-1) + j
                target_edges[0, k] = head
                target_edges[2, k] = ch[2]
    else:
        head = coref_nids[-2]
        for i in range(n_choices):
            ch = choices[i]
            target_edges[0, i] = head
            target_edges[2, i] = ch[2]
    target_edges = target_edges.unsqueeze(0)

    batch = {
        'bg': [gs],
        'input_ids': bert_inputs[0],
        'input_masks': bert_inputs[1],
        'token_type_ids': bert_inputs[2],
        'target_idxs': target_idxs,
        'nid2rows': [nid2rows],
        'n_instances': [bert_inputs.shape[1]],
        'target_edges': [target_edges]
    }
    with torch.no_grad():
        if args.gpu_id !=  -1:
            batch = NarrativeGraphDataset.to_gpu(batch, args.gpu_id)
        pred_scores, y = model(mode='predict', **batch)

        if score_pooling == 'mean':
            cand_scores = pred_scores.view(-1, chain_len-1).mean(1)
        elif score_pooling == 'max':
            cand_scores = pred_scores.view(-1, chain_len-1).max(1)[0]
        else: # last
            cand_scores = pred_scores
        pred = torch.argmax(cand_scores).cpu()

        if 'predict' not in results:
            results['predict'] = []
        if 'y' not in results:
            results['y'] = []

        logger.debug('scores={}, pred={}, y={}'.format(cand_scores, pred, correct))
        results['predict'].append(pred.item())
        results['y'].append(correct)


def eval_mcnc(ng_config, model):
    rtype2idx = ng_config['rtype2idx']
    idx2rtype = {v: k for k, v in rtype2idx.items()}

    coref_ridx = rtype2idx['cnext']

    t1 = time.time()
    results = {}
    count_gids = 0
    fs = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.h5')])
    for f in fs:
        fpath = os.path.join(args.input_dir, f)
        logger.info('processing {}...'.format(fpath))

        qf = '.'.join(f.split('.')[:-1])
        q_fpath = os.path.join(args.input_dir, 'q_{}.pkl'.format(qf))
        logger.info('loading {}...'.format(q_fpath))
        questions = pkl.load(open(q_fpath, 'rb'))

        t2 = time.time()
        fr = h5py.File(fpath, 'r')
        for gn in tqdm(fr.keys()):
            q = questions[gn]
            gid = int(gn.split('_')[-1])

            # load graph data
            bert_inputs = fr[gn]['bert_inputs'][:]
            bert_target_idxs = fr[gn]['bert_target_idxs'][:]
            bert_nid2rows = fr[gn]['bert_nid2rows'][:]
            ng_edges = fr[gn]['ng_edges'][:]
            coref_nids = fr[gn]['coref_nids'][:]
            test_data = {
                'ng_edges': ng_edges,
                'bert_inputs': bert_inputs,
                'bert_target_idxs': bert_target_idxs,
                'bert_nid2rows': bert_nid2rows,
                'coref_nids': coref_nids
            }

            # update results
            if args.model_name in ['lstm', 'mem']:
                eval_one_mcnc_question_lstm(model, test_data, results, q, coref_ridx, args.score_pooling)
            else:
                eval_one_mcnc_question(model, test_data, results, q, coref_ridx, args.score_pooling)

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
    acc = accuracy_score(results['y'], results['predict'])
    logger.info('#instances={}, accuracy={}'.format(len(results['y']), acc))


def main():
    config = json.load(open(args.ng_config))
    assert config["config_target"] == "narrative_graph"

    model = load_model(
        args.model_name, args.model_dir, args.model_config, args.gpu_id)
    model.eval()

    eval_mcnc(config, model)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
