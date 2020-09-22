import os
import re
import sys
import h5py
import time
import json
import logging
import argparse
import pickle as pkl
import multiprocessing
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

from nglib.common import utils
from nglib.corpora import GigawordNYT
from nglib.narrative.narrative_graph import NarrativeGraph, NGNode
from nglib.narrative.narrative_graph import get_next_relations
from nglib.narrative.narrative_graph import create_narrative_graph
from nglib.narrative.narrative_graph import get_ng_bert_tokenizer


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='prepare narrative graphs')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='config file of sentence sampling')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')

    parser.add_argument('--target_split', type=str, default=None,
                        choices=['train', 'dev', 'test'],
                        help='target split (train, dev, test)')
    parser.add_argument("--bert_weight_name",
                        default='google/bert_uncased_L-2_H-128_A-2', type=str,
                        help="bert weight version")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="max sequence length for BERT encoder")

    parser.add_argument("--min_coref_chain_len", default=9, type=int,
                        help="min coref chain len (default 9)")
    parser.add_argument("--instance_min", default=20, type=int,
                        help="minimum number of instances (default 20)")
    parser.add_argument("--instance_max", default=350, type=int,
                        help="maximum number of instances (default 350)")
    parser.add_argument('--is_cased', action='store_true', default=False,
                        help='BERT is case sensitive')
    parser.add_argument('--save_ng_pkl', action='store_true', default=False,
                        help='save narrative graph pickle (space-consuming)')
    parser.add_argument('--no_discourse', action='store_true', default=False,
                        help='no discourse relations')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def write_instance(
        cur_gid, doc_id, bert_inputs, rgcn_inputs, coref_chains, fw_h5, fw_chains):
    # rgcn_inputs
    edge_src = np.array(rgcn_inputs['edge_src'])
    edge_types = np.array(rgcn_inputs['edge_types'])
    edge_dest = np.array(rgcn_inputs['edge_dest'])
    edge_norms = np.array(rgcn_inputs['edge_norms'])
    ng_edges = np.vstack((edge_src, edge_types, edge_dest, edge_norms))
    fw_h5.create_dataset('graph_{}/ng_edges'.format(cur_gid), data=ng_edges)

    # bert_inputs
    binput = torch.stack((bert_inputs['input_ids'], bert_inputs['input_masks'], bert_inputs['token_type_ids']), dim=0)

    # turn nid2rows into numpy array with -1 padding
    max_len = 0
    nid2rows = bert_inputs['nid2rows']
    for nid in range(len(nid2rows)):
        idxs = nid2rows[nid]
        if len(idxs) > max_len:
            max_len = len(idxs)
    new_nid2rows = -1 * np.ones((len(nid2rows), max_len), dtype=np.int64)
    for nid in range(len(nid2rows)):
        idxs = nid2rows[nid]
        new_nid2rows[nid, :len(idxs)] = idxs

    fw_h5.create_dataset('graph_{}/bert_inputs'.format(cur_gid), data=binput.cpu().numpy())
    fw_h5.create_dataset('graph_{}/bert_target_idxs'.format(cur_gid), data=bert_inputs['target_idxs'].cpu().numpy())
    fw_h5.create_dataset('graph_{}/bert_nid2rows'.format(cur_gid), data=new_nid2rows)

    if len(coref_chains) > 0:
        # write h5
        h5_cchains = []
        for cchain in coref_chains:
            h5_cchain = [e['nid'] for e  in cchain]
            h5_cchains.append(h5_cchain)
        h5_cchains = chain2numpy(h5_cchains)
        fw_h5.create_dataset(
            'graph_{}/coref_chains'.format(cur_gid), data=h5_cchains)

        # write json
        out = {
            'doc_id': doc_id,
            'gid': cur_gid,
            'chains': coref_chains
        }
        tmp = json.dumps(out)
        fw_chains.write(tmp + '\n')


def chain2numpy(chains):
    m = max([len(c) for c in chains])
    arr = np.ones((len(chains), m), dtype=np.int64) * -1
    for i in range(len(chains)):
        arr[i][:len(chains[i])] = chains[i]
    return arr


def worker(pidx, g_graph_counts, g_rel_counts,
           src_fpath, target_dir, prefix,
           rtype2idx, dmarkers, no_entity):
    logger = utils.get_root_logger(args, log_fname='{}.log'.format(prefix))

    dest_fpath = os.path.join(target_dir, '{}.h5'.format(prefix))
    chain_fpath = os.path.join(target_dir, '{}.json'.format(prefix))
    docid_fpath = os.path.join(target_dir, '{}.txt'.format(prefix))

    logger.info('processing {} -> {}'.format(src_fpath, dest_fpath))
    bert_tokenizer = get_ng_bert_tokenizer(
        args.bert_weight_name, args.is_cased)

    t1 = time.time()
    gid2docid = OrderedDict()
    count_docs = 0
    count_valid = 0
    count_long_doc = 0

    fw_chains = open(chain_fpath, 'w')
    fw_h5 = h5py.File(dest_fpath, 'w')
    with open(src_fpath, 'r') as fr:
        for line in fr:
            doc = json.loads(line)

            count_docs += 1
            # if count_docs == 1000: #debug
            #     break
            if count_docs % 1000 == 0:
                logger.info('p{}: count_docs={}, {} s'.format(
                    pidx, count_docs, time.time()-t1))

            # filter by #nodes
            ng, coref_chains = create_narrative_graph(
                doc, rtypes=rtype2idx, dmarkers=dmarkers, no_entity=no_entity,
                min_coref_chain_len=args.min_coref_chain_len
            )
            if (len(ng.nodes) < args.instance_min
                    or len(ng.nodes) > args.instance_max):
                continue

            bert_inputs, rgcn_inputs = ng.to_dgl_inputs(
                bert_tokenizer, max_seq_len=args.max_seq_len)

            if bert_inputs is None:
                # sentence too long, skip this doc
                count_long_doc += 1
                logger.debug('p{}: sentence too long. skip the graph: {}'.format(
                    pidx, count_long_doc))
                continue

            if args.save_ng_pkl:
                # only use it for small corpus
                dpath = os.path.join(target_dir, 'ng_pickles')
                if not os.path.exists(dpath):
                    os.makedirs(dpath)
                fpath = os.path.join(dpath, '{}_{}.pkl'.format(
                    count_valid, doc['doc_id']))
                logger.debug('saving {}...'.format(fpath))
                pkl.dump(ng, open(fpath, 'wb'))

            gid2docid[count_valid] = doc['doc_id']
            write_instance(
                count_valid, doc['doc_id'], bert_inputs, rgcn_inputs,
                coref_chains, fw_h5, fw_chains)

            estats = ng.get_edge_stats()
            for rtype, c in estats.items():
                g_rel_counts[rtype2idx[rtype]] += c
            count_valid += 1

    fw_h5.close()
    fw_chains.close()

    g_graph_counts[pidx] = count_valid
    with open(docid_fpath, 'w') as fw:
        for gid, docid in gid2docid.items():
            fw.write('{}\t{}\n'.format(gid, docid))

    logger.info('skip {} docs, because of long sentences.'.format(count_long_doc))
    logger.info('{} / {} valid documents'.format(count_valid, count_docs))
    logger.info('dump graphs: {} s'.format(time.time()-t1))


def get_marker2rtype(rtype2markers):
    dmarkers = {}
    for rtype, markers in rtype2markers.items():
        for m in markers:
            dmarkers[m] = rtype
    return dmarkers


def main():
    config = json.load(open(args.config_file))
    assert config["config_target"] == "narrative_graph"
    rtype2idx = config['rtype2idx']
    if args.no_discourse:
        dmarkers = None
    else:
        dmarkers = get_marker2rtype(config['discourse_markers'])

    train_dir = os.path.join(config['nyt_dir'], 'train')
    dev_dir = os.path.join(config['nyt_dir'], 'dev')
    test_dir = os.path.join(config['nyt_dir'], 'test')
    if args.target_split is None:
        train_fs = sorted([os.path.join(train_dir, fn) for fn in os.listdir(train_dir)])
        dev_fs = sorted([os.path.join(dev_dir, fn) for fn in os.listdir(dev_dir)])
        test_fs = sorted([os.path.join(test_dir, fn) for fn in os.listdir(test_dir)])
        fs = train_fs + dev_fs + test_fs

    elif args.target_split == 'train':
        fs = sorted([os.path.join(train_dir, fn) for fn in os.listdir(train_dir)])
    elif args.target_split == 'dev':
        fs = sorted([os.path.join(dev_dir, fn) for fn in os.listdir(dev_dir)])
    elif args.target_split == 'test':
        fs = sorted([os.path.join(test_dir, fn) for fn in os.listdir(test_dir)])
    else:
        raise ValueError('target split: {}'.format(args.target_split))

    t1 = time.time()
    g_graph_counts = multiprocessing.Array('i', len(fs))

    all_g_rel_counts = []
    ps = []
    for pidx, src_fpath in enumerate(fs):
        g_rel_counts = multiprocessing.Array('i', len(rtype2idx))
        for i in range(len(rtype2idx)):
            g_rel_counts[i] = 0
        all_g_rel_counts.append(g_rel_counts)

        sp = src_fpath.split('/')
        fn = sp[-1]
        split = sp[-2]

        target_dir = os.path.join(args.output_dir, split)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        prefix = fn.split('.')[0]

        # worker(pidx,
        #        g_graph_counts,
        #        g_rel_counts,
        #        src_fpath, target_dir,
        #        prefix,
        #        rtype2idx,
        #        dmarkers,
        #        config['no_entity']) # debug
        p = multiprocessing.Process(target=worker,
                                    args=(pidx,
                                          g_graph_counts,
                                          g_rel_counts,
                                          src_fpath, target_dir,
                                          prefix,
                                          rtype2idx,
                                          dmarkers,
                                          config['no_entity'])
                                    )
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    # output stats
    fpath = os.path.join(args.output_dir, 'stats.txt')
    with open(fpath, 'w') as fw:
        text = 'all done: {} s, {} docs'.format(time.time()-t1, sum(g_graph_counts))
        print(text)
        fw.write(text+'\n')

        # merge all_g_rel_counts
        all_rel_counts = [0] * len(rtype2idx)
        for rcounts in all_g_rel_counts:
            for i in range(len(rtype2idx)):
                all_rel_counts[i] += rcounts[i]

        for rtype, idx in rtype2idx.items():
            text = '{}: {} rels'.format(rtype, all_rel_counts[idx])
            print(text)
            fw.write(text + '\n')


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    main()
