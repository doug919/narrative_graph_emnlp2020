import os
import csv
import sys
import time
import json
import h5py
import logging
import pickle as pkl
import argparse
import random
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

from nglib.common import utils
from nglib.narrative.dataset import sample_mcnc
from nglib.narrative.narrative_graph import get_pp_ridx2distr
from nglib.narrative.narrative_graph import get_pp_ridx2distr_coref


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='prepare coref chains for MCNC')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='ng config file')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input dir')
    parser.add_argument('prefix', metavar='PREFIX',
                        help='output file prefix')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')

    parser.add_argument('--is_training_set', action='store_true', default=False,
                        help='is training set')

    parser.add_argument("--chain_len", type=int, default=9,
                        help="fixed chain length (default: 9).")
    parser.add_argument("--n_choices", type=int, default=5,
                        help="number of choices (default: 5).")

    parser.add_argument("--n_per_file", type=int, default=5000,
                        help="number of graphs per file.")
    parser.add_argument("--sample_n_graphs", type=int, default=-1,
                        help="sample n graphs.")

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


def load_gid2docid(fpath):
    logger.info('loading {}...'.format(fpath))
    gid2docid = OrderedDict()
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.rstrip('\n')
            sp = line.split('\t')
            gid2docid[int(sp[0])] = sp[1]
    logger.info('{} documents loaded.'.format(len(gid2docid)))
    return gid2docid


def load_chain_info(fpath):
    logger.info('loading {}...'.format(fpath))
    out = OrderedDict()
    with open(fpath, 'r') as fr:
        for line in fr:
            d = json.loads(line)
            out[d['doc_id']] = d
    logger.info('{} documents loaded.'.format(len(out)))
    return out


def main():
    config = json.load(open(args.config_file))
    assert config["config_target"] == "narrative_graph"

    logger.warning('!!!!!!!!Note that NO_ENTITY={}. Please make sure it is correct'.format(config['no_entity']))

    rtype2idx = config['rtype2idx']
    coref_ridx = rtype2idx['cnext']

    new_gid = 0
    new_gid2docid = OrderedDict()

    t2 = time.time()
    fs = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.h5')])
    count_fw = 0
    questions = {}
    fw = os.path.join(args.output_dir, '{}_{}.h5'.format(args.prefix, count_fw))
    logger.info('writing {}...'.format(fw))
    fw = h5py.File(fw, 'w')
    for f in fs:
        fname = f.split('.')[0]
        fpath = os.path.join(args.input_dir, '{}.txt'.format(fname))
        gid2docid = load_gid2docid(fpath)

        fpath = os.path.join(args.input_dir, '{}.json'.format(fname))
        doc_chain_info = load_chain_info(fpath)

        fpath = os.path.join(args.input_dir, f)
        logger.info('processing {}...'.format(fpath))

        fr = h5py.File(fpath, 'r')
        for gn in tqdm(fr.keys()):
            gid = int(gn.split('_')[-1])
            docid = gid2docid[gid]
            if docid not in doc_chain_info:
                logger.debug('{} does not have coref chains'.format(docid))
                continue

            # read
            ng_edges = fr[gn]['ng_edges'][:]
            bert_inputs = fr[gn]['bert_inputs'][:]
            bert_target_idxs = fr[gn]['bert_target_idxs'][:]
            bert_nid2rows = fr[gn]['bert_nid2rows'][:]
            coref_chains = fr[gn]['coref_chains'][:]

            # fixed chain length
            chain_info = doc_chain_info[docid]['chains'][0][:args.chain_len]
            coref_nids = [m['nid'] for m in chain_info]

            if not args.is_training_set:
                # use only one chain for each graph
                n_nodes = bert_nid2rows.shape[0]

                # random sample
                correct, choices, target_edge_idx = sample_mcnc(
                    ng_edges, n_nodes, coref_nids, coref_ridx, args.n_choices)

                neg_nids = [choices[i][2] for i in range(args.n_choices) if i != correct]
                fw['graph_{}/negative_coref_nids'.format(new_gid)] = np.array(neg_nids, dtype=np.int64)

                questions['graph_{}'.format(new_gid)] = (correct, choices, target_edge_idx, chain_info)

            # dump output
            new_gid2docid[new_gid] = docid

            fw['graph_{}/bert_inputs'.format(new_gid)] = bert_inputs
            fw['graph_{}/bert_target_idxs'.format(new_gid)] = bert_target_idxs
            fw['graph_{}/bert_nid2rows'.format(new_gid)] = bert_nid2rows
            fw['graph_{}/ng_edges'.format(new_gid)] = ng_edges
            fw['graph_{}/coref_nids'.format(new_gid)] = np.array(coref_nids, dtype=np.int64)

            new_gid += 1
            if new_gid % args.n_per_file == 0:
                logger.info('sampling {} files takes {} s'.format(args.n_per_file, time.time()-t2))

                fw.close()
                if not args.is_training_set:
                    fpath = os.path.join(args.output_dir, 'q_{}_{}.pkl'.format(args.prefix, count_fw))
                    logger.info('writing {}...'.format(fpath))
                    pkl.dump(questions, open(fpath, 'wb'))

                # next file
                count_fw += 1
                fpath = os.path.join(args.output_dir, '{}_{}.h5'.format(args.prefix, count_fw))
                logger.info('writing {}...'.format(fpath))
                fw = h5py.File(fpath, 'w')

                questions = {}

                # restart timer
                t2 = time.time()
            if args.sample_n_graphs !=-1 and new_gid >= args.sample_n_graphs:
                break
        if args.sample_n_graphs != -1 and new_gid >= args.sample_n_graphs:
            break

        fr.close()
    fw.close()
    if not args.is_training_set:
        fpath = os.path.join(args.output_dir, '{}_{}.pkl'.format(args.prefix, count_fw))
        logger.info('writing {}...'.format(fpath))
        pkl.dump(questions, open(fpath, 'wb'))

    fpath = os.path.join(args.output_dir, 'gid2docid.txt')
    with open(fpath, 'w') as fw:
        for gid, did in new_gid2docid.items():
            fw.write('{}\t{}\n'.format(gid, did))

    logger.info('#graphs = {}'.format(len(new_gid2docid)))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
