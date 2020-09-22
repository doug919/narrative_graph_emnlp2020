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
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm, trange

from nglib.common import utils


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='more intrinsic evaluations')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='ng config file')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input dir')
    parser.add_argument('prefix', metavar='PREFIX',
                        help='output file prefix')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output dir')

    parser.add_argument("--n_choices", type=int, default=8,
                        help="number of choices")

    parser.add_argument("--seed", type=int, default=135,
                        help="random seed for initialization")

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
    # if gpu != -1:
    #     torch.cuda.manual_seed_all(seed)


def _get_indices_by_rtypes(ng_edges, rtype_idxs):
    n_edges = ng_edges.shape[1]
    ret_idxs = []
    for i in range(n_edges):
        e = tuple(ng_edges[:3, i].astype('int64'))
        if e[1] in rtype_idxs:
            ret_idxs.append(i)
    return ret_idxs


def _get_tail_node_repr_by_eidx(
        gid, cand_idx, ng_edges, bert_nid2rows, bert_inputs, bert_target_idxs):
    _edge = ng_edges[:, cand_idx] # outputs
    _nid = int(_edge[2])

    _row = bert_nid2rows[_nid]
    _row = _row[_row != -1]
    assert _row.shape == (1, ) # for pp links

    binputs = bert_inputs[:, _row, :].squeeze()
    target_col = bert_target_idxs[_row]
    # choice format
    ret = {
        'gid': gid,
        'eidx': cand_idx, # we need this because we might need to remove the edge
        'edge': _edge,
        'nid': _nid,
        'bert_inputs': binputs,
        'target_col': target_col
    }
    return ret


def sample_node_multiple_choice(ng_edges,
                                bert_inputs, bert_target_idxs, bert_nid2rows,
                                interested_rel_idxs, fr, gid):

    interested_eidxs = _get_indices_by_rtypes(ng_edges, interested_rel_idxs)
    if len(interested_eidxs) == 0:
        return None

    # sample a target node which is the tail node of the selected edge
    eidx = interested_eidxs[random.randint(0, len(interested_eidxs)-1)]
    answer = _get_tail_node_repr_by_eidx(
        gid, eidx, ng_edges, bert_nid2rows, bert_inputs, bert_target_idxs
    )

    target_e = ng_edges[:3, eidx].astype('int64')
    n_nodes = bert_nid2rows.shape[0]
    choices = [answer]
    gid_pool = [k for k in fr.keys()]
    while len(choices) < args.n_choices:
        # random a graph
        rgid = gid_pool[random.randint(0, len(gid_pool)-1)]
        rgid = int(rgid.split('_')[1])
        if rgid == gid:
            continue

        key = 'graph_{}'.format(rgid)
        r_binputs = fr[key]['bert_inputs'][:]
        r_target_idxs = fr[key]['bert_target_idxs'][:]
        r_nid2rows = fr[key]['bert_nid2rows'][:]
        r_ng_edges = fr[key]['ng_edges'][:]

        # sample a predicate node using the same manner as the answer
        interested_eidxs = _get_indices_by_rtypes(r_ng_edges, interested_rel_idxs)
        if len(interested_eidxs) == 0:
            continue

        eidx = interested_eidxs[random.randint(0, len(interested_eidxs)-1)]
        c = _get_tail_node_repr_by_eidx(
            rgid, eidx, r_ng_edges, r_nid2rows, r_binputs, r_target_idxs
        )
        choices.append(c)

    # shuffle choices
    choice_idxs = list(range(args.n_choices))
    random.shuffle(choice_idxs)
    correct = choice_idxs.index(0)
    choices = [choices[cidx] for cidx in choice_idxs]
    return (correct, choices)


def sample_node_multiple_choice_v2(
        ng_edges,
        bert_inputs, bert_target_idxs, bert_nid2rows,
        interested_rel_idxs, fr, gid):

    interested_eidxs = _get_indices_by_rtypes(ng_edges, interested_rel_idxs)
    if len(interested_eidxs) == 0:
        return None

    pos_edges = set()
    related_edges = ng_edges[:3, interested_eidxs].astype('int64')
    for i in range(related_edges.shape[1]):
        e = tuple(related_edges[:, i].flatten())
        pos_edges.add(e)

    eidx = interested_eidxs[random.randint(0, len(interested_eidxs)-1)]

    new_edges = np.concatenate((ng_edges[:, :eidx], ng_edges[:, eidx+1:]), axis=1)
    target_e = tuple(ng_edges[:3, eidx].astype('int64'))
    n_nodes = bert_nid2rows.shape[0]
    choices  = [target_e]
    while len(choices) < args.n_choices:
        # try in-doc sampling

        # random a node
        r_nid = random.randint(0, n_nodes-1) # we don't separate entity or predicate
        r_e = (target_e[0], target_e[1], r_nid)
        if r_e in pos_edges:
            continue
        choices.append(r_e)

    # shuffle choices
    choice_idxs = list(range(args.n_choices))
    random.shuffle(choice_idxs)
    correct = choice_idxs.index(0)
    choices = [choices[cidx] for cidx in choice_idxs]
    return (new_edges, correct, choices)


def sample_ep_questions(ng_edges, rtype2idx):
    # join ep_edges by src node
    src_nodes = {}
    ep_ridxs = {rtype2idx['s'], rtype2idx['o'], rtype2idx['prep']}
    n_edges = ng_edges.shape[1]

    all_pos_edges = set()
    entity_nids = set()
    for i in range(n_edges):
        e = tuple(ng_edges[:3, i].astype('int64'))
        all_pos_edges.add(e)
        if e[1] in ep_ridxs:
            if e[0] not in src_nodes:
                src_nodes[e[0]] = []
            src_nodes[e[0]].append((i, e))
            entity_nids.add(int(e[2]))

    if len(entity_nids) < args.n_choices:
        return None, None

    # Task1, random a event node with 3 ep edges, predict edge types
    candidate_sources = {src: es for src, es in src_nodes.items() if len(es) >= 3}
    if len(candidate_sources) == 0:
        return None, None

    keys = list(candidate_sources.keys())
    src_nid = keys[random.randint(0, len(candidate_sources)-1)]
    edges = candidate_sources[src_nid]
    q_link = (src_nid, edges) # question

    # Task2, random one edge, predict one entity
    entity_nid_list = list(entity_nids)
    r_eidx, r_edge = edges[random.randint(0, len(edges)-1)]
    answer = r_edge[2]

    choices = [answer]
    while len(choices) < args.n_choices:
        r_nid = entity_nid_list[random.randint(0, len(entity_nid_list)-1)]
        r_tri = (r_edge[0], r_edge[1], r_nid)
        if r_tri in all_pos_edges:
            continue
        choices.append(r_nid)

    idxs = list(range(len(choices)))
    random.shuffle(idxs)
    correct = idxs.index(0)
    choices = [choices[i] for i in idxs]
    q_entity = (r_eidx, r_edge, correct, choices) # question
    return q_link, q_entity


def main():
    config = json.load(open(args.config_file))
    assert config["config_target"] == "narrative_graph"

    rtype2idx = config['rtype2idx']
    if config['no_entity']:
        ep_rtype_rev = {}
        ent_pred_ridxs = set()
    else:
        ep_rtype_rev = {rtype2idx[v]: rtype2idx[k] for k, v in
                              config['entity_predicate_rtypes'].items()}
        ent_pred_ridxs = set(ep_rtype_rev.keys())
    n_rtypes = len(rtype2idx)
    pred_pred_ridxs = set(range(n_rtypes)) - ent_pred_ridxs
    disc_pred_pred_ridxs = pred_pred_ridxs - {rtype2idx['next'], rtype2idx['cnext']}

    t2 = time.time()
    q_counts = {
        'pp_coref_next': 0,
        'pp_next': 0,
        'pp_discourse_next': {},
        'ep_link': 0,
        'ep_entity': {}
    }
    count_gids = 0
    fs = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.h5')])
    for f in fs:
        fpath = os.path.join(args.input_dir, f)
        logger.info('processing {}...'.format(fpath))

        fr = h5py.File(fpath, 'r')
        questions = OrderedDict()
        for gn in tqdm(fr.keys()):
            questions[gn] = {}
            gid = int(gn.split('_')[-1])

            bert_inputs = fr[gn]['bert_inputs'][:]
            bert_target_idxs = fr[gn]['bert_target_idxs'][:]
            bert_nid2rows = fr[gn]['bert_nid2rows'][:]
            ng_edges = fr[gn]['ng_edges'][:]
            n_nodes = bert_nid2rows.shape[0]

            # # sample PP_COREF_NEXT task
            q = sample_node_multiple_choice_v2(
                    ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows,
                    {rtype2idx['cnext']}, fr, gid)
            questions[gn]['pp_coref_next'] = q
            if q is not None:
                q_counts['pp_coref_next'] += 1


            # sample PP_NEXT task
            q = sample_node_multiple_choice_v2(
                    ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows,
                    {rtype2idx['next']}, fr, gid)
            questions[gn]['pp_next'] = q
            if q is not None:
                q_counts['pp_next'] += 1


            # sample PP_DISCOURSE_NEXT task
            q = sample_node_multiple_choice_v2(
                    ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows,
                    disc_pred_pred_ridxs, fr, gid)
            questions[gn]['pp_discourse_next'] = q
            if q is not None: # count by rtypes
                ans = q[2][q[1]]
                rtype = ans[1]
                # ans = q[1][q[0]]
                # rtype = int(ans['edge'][1])
                if rtype not in q_counts['pp_discourse_next']:
                    q_counts['pp_discourse_next'][rtype] = 0
                q_counts['pp_discourse_next'][rtype] += 1


            # sample PP_DISCOURSE_LINK_TYPE task
            # reuse the above links

            # sample PP_DISCOURSE_TRIPLET task
            # evaluate on the sampled test set

            if not config['no_entity']:
                # sample EP_LINK_TYPE
                q_link, q_entity = sample_ep_questions(ng_edges, rtype2idx)
                questions[gn]['ep_link'] = q_link
                if q_link is not None:
                    q_counts['ep_link'] += 1

                # # sample EP_NODE task
                questions[gn]['ep_entity'] = q_entity
                if q_entity is not None:
                    rtype = int(q_entity[1][1])
                    if rtype not in q_counts['ep_entity']:
                        q_counts['ep_entity'][rtype] = 0
                    q_counts['ep_entity'][rtype] += 1

            count_gids += 1
        fr.close()

        # dump questions for a file
        fn = '.'.join(f.split('.')[:-1])
        fpath = os.path.join(args.output_dir, 'q_{}.pkl'.format(fn))
        logger.info('dumping {}...'.format(fpath))
        pkl.dump(questions, open(fpath, 'wb'))

    logger.info('#graphs = {}'.format(count_gids))
    logger.info('q_counts = {}'.format(q_counts))


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
