import os
import argparse
import logging
import time
import json
import codecs
from collections import OrderedDict
import random
from copy import deepcopy
import re

import dgl
import h5py
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from allennlp.modules.elmo import Elmo, batch_to_ids

from nglib.common import utils
from nglib.common import discourse as ds
from nglib.narrative.dataset import NarrativeGraphDataset
from nglib.narrative.dataset import _update_norms
from nglib.narrative.narrative_graph import create_narrative_graph
from nglib.narrative.narrative_graph import pad_nid2rows
from nglib.narrative.narrative_graph import get_ng_bert_tokenizer
from nglib.models.bert_rgcn import RGCNLinkPredict


elmo_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
elmo_option_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='preprocess CONLL2016')
    parser.add_argument('ds_file', metavar='DS_FILE',
                        help='the event file for Discourse Sense')
    parser.add_argument('parse_file', metavar='PARSE_FILE',
                        help='the parse file for Discourse Sense')
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER',
                        help='the folder for outputs.')
    # parser.add_argument('feature_set', metavar='feature_set', choices=['elmo', 'ng'],
    #                     help='feature set')
    parser.add_argument("--bert_weight_name",
                        default='google/bert_uncased_L-2_H-128_A-2', type=str,
                        help="bert weight version")
    parser.add_argument('--is_cased', action='store_true', default=False,
                        help='BERT is case sensitive')
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="max sequence length for BERT encoder")
    parser.add_argument('--ng_model_dir', default=None, type=str,
                        help='checkpoint directory')
    parser.add_argument('--ng_model_config', default=None, type=str,
                        help='ng_model config')
    parser.add_argument('--ng_config', default=None, type=str,
                        help='ng_config')

    parser.add_argument('--max_event_seq_len', type=int, default=None,
                        help='max_event_seq_len')
    parser.add_argument('--no_parse', action='store_true', default=False,
                        help='no parse file')
    parser.add_argument('--no_bert_emb', action='store_true', default=False,
                        help='not use bert embeddings')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')

    args = parser.parse_args(argv)
    return args


def get_max_arg_length(rels):
    ret = 0
    for rel in rels:
        arg1_len = len(rel['Arg1']['TokenList'])
        arg2_len = len(rel['Arg2']['TokenList'])

        if arg1_len > ret:
            ret = arg1_len
        if arg2_len > ret:
            ret = arg2_len
    return ret


def get_arg_nids(rel, arg_key, loc2nid):
    arg_nids = []
    for tok in rel[arg_key]['TokenList']:
        sent_idx, tidx = tok[3], tok[4]
        if sent_idx in loc2nid:
            if tidx in loc2nid[sent_idx]:
                arg_nids.append(loc2nid[sent_idx][tidx])
    return arg_nids


def get_max_rel_ng_nodes(rels, ngs):
    ret = 0
    for rel in rels:
        doc_id = rel['DocID']
        if doc_id not in ngs:
            logger.info('rel {} skipped'.format(rel['ID']))
            continue
        ng, emb, bert_inputs, rgcn_inputs = ngs[doc_id]
        loc2nid = ng.get_loc2nid()

        arg1_nids = get_arg_nids(rel, 'Arg1', loc2nid)
        arg2_nids = get_arg_nids(rel, 'Arg2', loc2nid)

        if len(arg1_nids) > ret:
            ret = len(arg1_nids)
        if len(arg2_nids) > ret:
            ret = len(arg2_nids)
    return ret


def get_arg_tokens(rel, arg_key, parses):
    if parses is None:
        text = rel[arg_key]['RawText']
        toks = text.split(' ')
    else:
        token_list = rel[arg_key]['TokenList']
        locs = [(tok[3], tok[4]) for tok in token_list]

        parse = parses[rel['DocID']]
        toks = [parse['sentences'][loc[0]]['tokens'][loc[1]]['word'] for loc in locs]
    return toks


def elmo_features(elmo, rels, rel2idx, parses, seq_len=None, gpu_id=None):
    with torch.no_grad():
        arg0s = [get_arg_tokens(rel, 'Arg1', parses) for rel in rels]
        char_ids = batch_to_ids(arg0s)
        if args.gpu_id is not None:
            char_ids = char_ids.cuda(gpu_id)
        a0we = elmo(char_ids)['elmo_representations'][0]

        arg1s = [get_arg_tokens(rel, 'Arg2', parses) for rel in rels]
        char_ids = batch_to_ids(arg1s)
        if args.gpu_id is not None:
            char_ids = char_ids.cuda(gpu_id)
        a1we = elmo(char_ids)['elmo_representations'][0]
        ys = torch.LongTensor([rel2idx[rel['Sense'][0]] for rel in rels])

        if seq_len:
            if a0we.shape[1] > seq_len:
                a0we = a0we[:, :seq_len, :]
            elif a0we.shape[1] < seq_len:
                dims = (a0we.shape[0], seq_len, a0we.shape[2])
                tmp = torch.zeros(dims, dtype=torch.float32)
                tmp[:, :a0we.shape[1]] = a0we
                a0we = tmp
            if a1we.shape[1] > seq_len:
                a1we = a1we[:, :seq_len, :]
            elif a1we.shape[1] < seq_len:
                dims = (a1we.shape[0], seq_len, a1we.shape[2])
                tmp = torch.zeros(dims, dtype=torch.float32)
                tmp[:, :a1we.shape[1]] = a1we
                a1we = tmp
    return a0we, a1we, ys


def build_target_edges(rgcn_inputs, arg1_nids, arg2_nids, rtype2idx):
    es = []
    for rtype, ridx in rtype2idx.items():
        for nid1 in arg1_nids:
            for nid2 in arg2_nids:
                e = [nid1, ridx, nid2, 1]
                es.append(e)

    if len(es) == 0:
        target_edges = None
    elif len(es) == 1:
        target_edges = torch.LongTensor(es).view(-1, 1)
    else:
        target_edges = torch.transpose(torch.LongTensor(es), 0, 1)
    return target_edges


def ng_features(ng_model, ngs, batch_rels, rel2idx, max_rel_ng_len, ng_config):
    rtype2idx = ng_config['rtype2idx']

    ys = torch.LongTensor([rel2idx[rel['Sense'][0]] for rel in batch_rels])

    h_dim = ng_model.rgcn.h_dim if args.no_bert_emb else ng_model.rgcn.h_dim*2
    x0_ng = torch.zeros((len(batch_rels), max_rel_ng_len, h_dim),
                        dtype=torch.float32)
    x1_ng = torch.zeros((len(batch_rels), max_rel_ng_len, h_dim),
                        dtype=torch.float32)
    ng_scores = []
    for i_rel, rel in enumerate(batch_rels):
        doc_id = rel['DocID']
        if doc_id not in ngs:
            logger.info('rel {} skipped'.format(rel['ID']))
            s = torch.zeros(len(rtype2idx))
            ng_scores.append(s)
            continue
        ng, embs, bert_inputs, rgcn_inputs = ngs[doc_id]
        loc2nid = ng.get_loc2nid()

        arg1_nids = sorted(get_arg_nids(rel, 'Arg1', loc2nid))
        arg2_nids = sorted(get_arg_nids(rel, 'Arg2', loc2nid))

        arg1_embs = [embs[nid] for nid in arg1_nids]
        arg2_embs = [embs[nid] for nid in arg2_nids]

        for i_emb, emb in enumerate(arg1_embs):
            x0_ng[i_rel][i_emb] = emb
        for i_emb, emb in enumerate(arg2_embs):
            x1_ng[i_rel][i_emb] = emb

        model_inputs = prepare_ng_model_inputs(bert_inputs, rgcn_inputs)
        # add target_edges
        target_edges = build_target_edges(
            rgcn_inputs, arg1_nids, arg2_nids, rtype2idx)
        if target_edges is not None:
            model_inputs['target_edges'] = [target_edges.unsqueeze(0)]
            if args.gpu_id is not None:
                model_inputs = NarrativeGraphDataset.to_gpu(model_inputs, args.gpu_id)

            with torch.no_grad():
                pred_scores, y = ng_model(**model_inputs, mode='predict')

            pred_scores = pred_scores.cpu()
            n_epairs = int(target_edges.shape[1] / len(rtype2idx))
            s = pred_scores.view(len(rtype2idx), -1).sum(1) / n_epairs
        else:
            s = torch.zeros(len(rtype2idx))
        ng_scores.append(s)

    ng_scores = torch.stack(ng_scores, dim=0)
    return x0_ng, x1_ng, ng_scores, ys


def dump_features(ds_fpath, elmo, parses, ng_model, ngs, ng_config, batch_size):
    rels = [json.loads(line) for line in open(ds_fpath)]
    # Non-Explicit = Implicit, EntRel, AltLex
    rels = [rel for rel in rels if rel['Type'] != 'Explicit'
            and rel['Sense'][0] in ds.CONLL16_REL2IDX]

    max_arg_len = get_max_arg_length(rels)
    max_rel_ng_len = get_max_rel_ng_nodes(rels, ngs)
    logger.info("max_arg_len={}, max_rel_ng_len={}".format(
        max_arg_len, max_rel_ng_len))
    n_batches = len(rels) // batch_size
    if len(rels) % batch_size != 0:
        n_batches += 1
    logger.info('#rels={}, batch_size={}, #batches={}'.format(len(rels), batch_size, n_batches))

    fpath = os.path.join(args.output_folder, 'data.h5')
    fw_h5 = h5py.File(fpath, 'w')

    y_out = fw_h5.create_dataset("y", data=np.zeros(len(rels), dtype=np.int64))
    # Feature: ELMo
    dim = 512 # elmo
    x0_elmo_out = fw_h5.create_dataset("x0_elmo", data=np.zeros((len(rels), max_arg_len, dim), dtype=np.float32))
    x1_elmo_out = fw_h5.create_dataset("x1_elmo", data=np.zeros((len(rels), max_arg_len, dim), dtype=np.float32))
    # Feature: NG
    x_ng_score_out = fw_h5.create_dataset(
        "x_ng_scores",
        data=np.zeros((len(rels), len(ng_config['rtype2idx'])), dtype=np.float32)
    )

    h_dim = ng_model.rgcn.h_dim if args.no_bert_emb else ng_model.rgcn.h_dim*2
    x0_ng_out = fw_h5.create_dataset(
        "x0_ng",
        data=np.zeros((len(rels), max_rel_ng_len, h_dim), dtype=np.float32)
    )
    x1_ng_out = fw_h5.create_dataset(
        "x1_ng",
        data=np.zeros((len(rels), max_rel_ng_len, h_dim), dtype=np.float32)
    )

    for i_batch in tqdm(range(n_batches)):
        start = i_batch * batch_size
        end = (i_batch+1) * batch_size
        batch_rels = rels[start:end]
        # elmo
        x0_elmo, x1_elmo, y_elmo = elmo_features(elmo, batch_rels, ds.CONLL16_REL2IDX, parses, gpu_id=args.gpu_id)
        # write to file
        y_out[start:end] = y_elmo.cpu().numpy()
        x0_elmo_out[start:end, :x0_elmo.shape[1]] = x0_elmo.cpu().numpy()
        x1_elmo_out[start:end, :x1_elmo.shape[1]] = x1_elmo.cpu().numpy()

        # NG features
        x0_ng, x1_ng, ng_scores, y_ng = ng_features(
            ng_model, ngs, batch_rels, ds.CONLL16_REL2IDX, max_rel_ng_len, ng_config)

        x_ng_score_out[start:end] = ng_scores.cpu().numpy()
        x0_ng_out[start:end] = x0_ng.cpu().numpy()
        x1_ng_out[start:end] = x1_ng.cpu().numpy()

    fw_h5.close()


def prepare_ng_model_inputs(bert_inputs, rgcn_inputs):
    edge_src = torch.LongTensor(rgcn_inputs['edge_src'])
    edge_types = torch.LongTensor(rgcn_inputs['edge_types'])
    edge_dest = torch.LongTensor(rgcn_inputs['edge_dest'])
    # back_edge_norms = torch.FloatTensor(rgcn_inputs['edge_norms']).unsqueeze(1)

    input_ids = bert_inputs['input_ids']
    input_masks = bert_inputs['input_masks']
    token_type_ids = bert_inputs['token_type_ids']
    target_idxs = bert_inputs['target_idxs']

    nid2rows = torch.from_numpy(pad_nid2rows(bert_inputs['nid2rows']))

    n_nodes = nid2rows.shape[0]
    n_instances = [input_ids.shape[0]]

    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    g.add_edges(edge_src, edge_dest)
    edge_norms = []
    for i in range(edge_dest.shape[0]) :
        nid2 = int(edge_dest[i])
        edge_norms.append(1.0 / g.in_degree(nid2))
    edge_norms = torch.FloatTensor(edge_norms).unsqueeze(1)
    g.edata.update({'rel_type': edge_types})
    g.edata.update({'norm': edge_norms})

    inputs = {
        'bg': [[g]],
        'input_ids': input_ids,
        'input_masks': input_masks,
        'token_type_ids': token_type_ids,
        'target_idxs': target_idxs,
        'nid2rows': [nid2rows],
        'n_instances': n_instances
    }
    return inputs


def get_ngs(model, parses, dmarkers, rtype2idx, no_entity):
    tokenizer = get_ng_bert_tokenizer(
        args.bert_weight_name, args.is_cased)
    t1 = time.time()
    ngs = {}
    n_skipped = 0
    for doc_id, parse in tqdm(parses.items(), desc='get NGs'):
        ng = create_narrative_graph(
            parse, dmarkers=dmarkers, rtypes=rtype2idx, no_entity=no_entity)
        # logger.info('#nodes = {}'.format(len(ng.nodes)))

        bert_inputs, rgcn_inputs = \
            ng.to_dgl_inputs(tokenizer, args.max_seq_len)
        # assert bert_inputs is not None, 'sentence too long'
        if bert_inputs is None or len(rgcn_inputs['edge_src']) == 0:
            logger.warning('{} has no edges or long sentences. skip.'.format(doc_id))
            n_skipped += 1
            continue

        model_inputs = prepare_ng_model_inputs(bert_inputs, rgcn_inputs)
        if args.gpu_id is not None:
            model_inputs = NarrativeGraphDataset.to_gpu(model_inputs, args.gpu_id)

        with torch.no_grad():
            rgcn_embs, bert_embs = model(**model_inputs, mode='embeddings')
        rgcn_emb = rgcn_embs[0].cpu()
        bert_emb = bert_embs[0].cpu()
        if args.no_bert_emb:
            emb = rgcn_emb
        else:
            emb = torch.cat((rgcn_emb, bert_emb), dim=1)
        ngs[doc_id] = (ng, emb, bert_inputs, rgcn_inputs)
    logger.info('get NG embeddings: {} s'.format(time.time()-t1))
    logger.info('#skipped={}'.format(n_skipped))
    return ngs


def load_ng_model(model_dir, model_config):
    # search models
    target_model_dir = utils.get_target_model_dir(model_dir)

    # load model
    logger.info('loading test model {}...'.format(target_model_dir))
    model_config = json.load(open(model_config, 'r'))
    model = RGCNLinkPredict.from_pretrained(target_model_dir, model_config)
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)
    return model


def get_marker2rtype(rtype2markers):
    dmarkers = {}
    for rtype, markers in rtype2markers.items():
        for m in markers:
            dmarkers[m] = rtype
    return dmarkers


def main():
    elmo = Elmo(elmo_option_file, elmo_weight_file, 1, dropout=0)
    elmo.eval()
    if args.gpu_id is not None:
        elmo.cuda(args.gpu_id)

    if args.no_parse:
        logger.warning('no parse file for tokenization')
        parses = None
    else:
        logger.info('loading parse from {}...'.format(args.parse_file))
        parses = utils.load_lined_json(args.parse_file, 'doc_id')

    ng_config = json.load(open(args.ng_config, 'r'))
    ng_model = load_ng_model(args.ng_model_dir, args.ng_model_config)
    dmarkers = get_marker2rtype(ng_config['discourse_markers'])
    ngs = get_ngs(ng_model, parses, dmarkers, ng_config['rtype2idx'], ng_config['no_entity'])

    dump_features(args.ds_file, elmo, parses, ng_model, ngs, ng_config, batch_size=args.batch_size)


if __name__ == '__main__':
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
