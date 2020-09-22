import os
import csv
import sys
import time
import json
import h5py
import logging
import argparse
import random
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

from nglib.common import utils
from nglib.narrative.dataset import sample_truncated_ng
from nglib.narrative.narrative_graph import get_pp_ridx2distr
from nglib.narrative.narrative_graph import get_pp_ridx2distr_coref


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='re-organize datasets and sample target edges if needed')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='ng config file')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input dir')
    parser.add_argument('prefix', metavar='PREFIX', choices=['train', 'dev', 'test'],
                        help='output file prefix')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')

    parser.add_argument('--not_sample_target_edges', action='store_true', default=False,
                        help='do not sample target edges')
    parser.add_argument('--sample_entity_only', action='store_true', default=False,
                        help='do not sample relation type')
    parser.add_argument('--sample_coref_only', action='store_true', default=False,
                        help='sample coref rel only')

    parser.add_argument("--n_truncated_ng", default=4, type=int,
                        help="number of truncated graphs for each NG.")
    parser.add_argument("--edge_sample_rate", default=0.05, type=float,
                        help="proportion of missing edges for each NG.")
    parser.add_argument("--n_neg_per_pos", default=20, type=int,
                        help="number of negative samles per positive edge.")

    parser.add_argument("--n_per_file", type=int, default=5000,
                        help="number of graphs per file.")

    parser.add_argument("--seed", type=int, default=135,
                        help="random seed for initialization")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if gpu != -1:
    #     torch.cuda.manual_seed_all(seed)


def load_gid2docid(fpath):
    gid2docid = OrderedDict()
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.rstrip('\n')
            sp = line.split('\t')
            gid2docid[int(sp[0])] = sp[1]
    return gid2docid


def main():
    config = json.load(open(args.config_file))
    assert config["config_target"] == "narrative_graph"

    logger.warning('!!!!!!!!Note that NO_ENTITY={}. Please make sure it is correct'.format(config['no_entity']))

    rtype2idx = config['rtype2idx']
    if args.sample_coref_only:
        pp_ridx2distr = get_pp_ridx2distr_coref(config)
    else:
        pp_ridx2distr = get_pp_ridx2distr(config)
    coref_ridx = rtype2idx['cnext']

    if config['no_entity']:
        ep_rtype_rev = {}
        ent_pred_ridxs = set()
    else:
        ep_rtype_rev = {rtype2idx[v]: rtype2idx[k] for k, v in
                              config['entity_predicate_rtypes'].items()}
        ent_pred_ridxs = set(ep_rtype_rev.keys())

    n_rtypes = len(rtype2idx)
    pred_pred_ridxs = set(range(n_rtypes)) - ent_pred_ridxs

    new_gid = 0
    n_truncated_graphs = 0
    n_target_edges = 0
    new_gid2docid = OrderedDict()

    t2 = time.time()
    fs = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.h5')])
    count_fw = 0
    fw = os.path.join(args.output_dir, '{}_{}.h5'.format(args.prefix, count_fw))
    logger.info('writing {}...'.format(fw))
    fw = h5py.File(fw, 'w')
    for f in fs:
        fname = f.split('.')[0]
        fpath = os.path.join(args.input_dir, '{}.txt'.format(fname))
        gid2docid = load_gid2docid(fpath)

        fpath = os.path.join(args.input_dir, f)
        logger.info('processing {}...'.format(fpath))

        fr = h5py.File(fpath, 'r')
        for gn in tqdm(fr.keys()):
            gid = int(gn.split('_')[-1])
            docid = gid2docid[gid]

            # read
            ng_edges = fr[gn]['ng_edges'][:]
            bert_inputs = fr[gn]['bert_inputs'][:]
            bert_target_idxs = fr[gn]['bert_target_idxs'][:]
            bert_nid2rows = fr[gn]['bert_nid2rows'][:]
            coref_chains = fr[gn]['coref_chains'][:] if 'coref_chains' in fr[gn] \
                else None

            # if the graph has tiny amount of edges we drop it.
            n_edges = ng_edges.shape[1]
            n_pp_edges = 0
            n_coref_edges = 0
            for i in range(n_edges):
                ridx = ng_edges[1, i].item()
                if ridx == coref_ridx:
                    n_coref_edges += 1
                if ridx in pred_pred_ridxs:
                    n_pp_edges += 1
            n_sampled_edges = int(n_pp_edges * args.edge_sample_rate)
            if n_sampled_edges == 0:
                logger.info('drop small graph: {}'.format(docid))
                continue
            if n_coref_edges == 0:
                logger.info('no coref edges: {}'.format(docid))
                continue

            # sample first
            if not args.not_sample_target_edges:
                n_nodes = bert_nid2rows.shape[0]
                ng_edges = torch.from_numpy(ng_edges.astype('float32'))
                all_target_edges, all_input_edges = \
                    sample_truncated_ng(ng_edges,
                                        n_nodes,
                                        ent_pred_ridxs,
                                        pred_pred_ridxs,
                                        ep_rtype_rev,
                                        args.n_truncated_ng,
                                        args.edge_sample_rate,
                                        args.n_neg_per_pos,
                                        pp_ridx2distr,
                                        coref_ridx,
                                        sample_entity_only=args.sample_entity_only)
                if all_target_edges is None:
                    logger.warning('no target edges sampled. drop it.')
                    continue

                all_target_edges = all_target_edges.numpy()
                all_input_edges = all_input_edges.numpy()

                fw['graph_{}/target_edges'.format(new_gid)] = all_target_edges
                fw['graph_{}/input_edges'.format(new_gid)] = all_input_edges

                n_truncated_graphs += all_target_edges.shape[0]
                n_target_edges += all_target_edges.shape[0] * all_target_edges.shape[2]

            new_gid2docid[new_gid] = docid

            fw['graph_{}/bert_inputs'.format(new_gid)] = bert_inputs
            fw['graph_{}/bert_target_idxs'.format(new_gid)] = bert_target_idxs
            fw['graph_{}/bert_nid2rows'.format(new_gid)] = bert_nid2rows
            fw['graph_{}/ng_edges'.format(new_gid)] = ng_edges
            if coref_chains is not None:
                fw['graph_{}/coref_chains'.format(new_gid)] = coref_chains

            new_gid += 1
            if new_gid % args.n_per_file == 0:
                logger.info('sampling {} files takes {} s'.format(args.n_per_file, time.time()-t2))

                fw.close()

                count_fw += 1
                fw = os.path.join(args.output_dir, '{}_{}.h5'.format(args.prefix, count_fw))
                logger.info('writing {}...'.format(fw))
                fw = h5py.File(fw, 'w')

                # restart timer
                t2 = time.time()

        fr.close()
    fw.close()

    fpath = os.path.join(args.output_dir, 'gid2docid.txt')
    with open(fpath, 'w') as fw:
        for gid, did in new_gid2docid.items():
            fw.write('{}\t{}\n'.format(gid, did))

    logger.info('#graphs = {}'.format(len(new_gid2docid)))
    logger.info('#truncated_graphs = {}'.format(n_truncated_graphs))
    logger.info('#target_edges = {}'.format(n_target_edges))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
