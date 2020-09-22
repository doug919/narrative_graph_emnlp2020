import sys
import os
import logging
import argparse
import json
import time
import re
import random
import pickle as pkl

import h5py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from nglib.common import utils
from nglib.common import discourse as ds
from nglib.narrative.narrative_graph import NarrativeGraph, NGNode
from nglib.narrative.narrative_graph import create_narrative_graph


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='training for Implicit Discourse Sense task')
    parser.add_argument('output_folder', metavar='OUTPUT_FOLDER',
                        help='output folder')
    parser.add_argument('mode', type=str, choices=['train', 'test'],
                        help='train or test')

    # train
    parser.add_argument('--train_dir', metavar='TRAIN_DIR', default=None,
                        help='preprocessed training data')
    parser.add_argument('--dev_dir', metavar='DEV_DIR', default=None,
                        help='preprocessed dev data')
    parser.add_argument('--dev_rel', metavar='DEV_REL', default=None,
                        help='dev rel json')

    # test
    parser.add_argument('--test_dir', metavar='TEST_DIR', default=None,
                        help='preprocessed test data')
    parser.add_argument('--test_rel', metavar='TEST_REL', default=None,
                        help='dev test json')
    parser.add_argument('--model_file', metavar='MODEL_FILE', default=None,
                        help='model file')


    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.95, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--optim", type=str, default='adamw', choices=['adamw', 'adagrad'],
                        help="optimizer")
    parser.add_argument("--seed", type=int, default=19777,
                        help="random seed for initialization")
    parser.add_argument('-g', '--gpu_id', type=int, default=None,
                        help='gpu id')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--train_batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('-e', '--max_n_epoches', type=int, default=500,
                        help='number of epoches')
    parser.add_argument('-p', '--patience', type=int, default=None,
                        help='patience for stopping')
    parser.add_argument('-r', '--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--use_ng', action='store_true', default=False,
                        help='use ng features')
    parser.add_argument('--use_ng_scores', action='store_true', default=False,
                        help='use ng_scores features')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='use class weights')

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


class CONLL16Dataset(Dataset):
    def __init__(self, fld, use_ng_features=True):
        super(CONLL16Dataset, self).__init__()
        self.fld = fld
        fpath = os.path.join(fld, 'data.h5')
        self.h5_file = h5py.File(fpath, 'r')
        self.x0 = self.h5_file.get('x0_elmo')
        self.x1 = self.h5_file.get('x1_elmo')
        self.y = self.h5_file.get('y')

        self._len = self.y.shape[0]
        assert self.x0.shape[0] == self._len
        assert self.x1.shape[0] == self._len
        self.seq_len = self.x0[0].shape[0]

        self.use_ng_features = use_ng_features
        if use_ng_features:
            self.x0_ng = self.h5_file.get('x0_ng')
            self.x1_ng = self.h5_file.get('x1_ng')
            assert self.x0_ng.shape == self.x1_ng.shape
            self.ng_seq_len = self.x0_ng[0].shape[0]
            self.x_ng_scores = self.h5_file.get('x_ng_scores')
            assert self.x0_ng.shape[0] == self._len
            assert self.x_ng_scores.shape[0] == self._len

    def __getitem__(self, idx):
        x0 = torch.from_numpy(self.x0[idx])
        x1 = torch.from_numpy(self.x1[idx])
        y = torch.LongTensor([self.y[idx]])

        if self.use_ng_features:
            x0_ng = self.x0_ng[idx]
            x1_ng = self.x1_ng[idx]
            x_ng = self.x_ng_scores[idx]
            return (x0, x1, x0_ng, x1_ng, x_ng), y
        return (x0, x1), y

    def __len__(self):
        return self._len


class AttentionNN(torch.nn.Module):
    def __init__(self, n_rel_classes, arg_dim=512, dropout=0.0,
                 ng_dim=256, ng_score_dim=8, use_ng=True, use_ng_scores=True,
                 use_class_weights=False):
        super(AttentionNN, self).__init__()
        self.dropout = dropout
        self.arg_dim = arg_dim
        self.n_rel_classes = n_rel_classes
        self.ng_dim = ng_dim
        self.ng_score_dim = ng_score_dim
        self.use_ng = use_ng
        self.use_ng_scores = use_ng_scores

        self.arg_attn = torch.nn.Parameter(torch.FloatTensor(2, arg_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.arg_attn.data)
        self.l1_0 = torch.nn.Linear(arg_dim, arg_dim//2)
        self.l1_1 = torch.nn.Linear(arg_dim, arg_dim//2)
        if use_ng:
            self.ng_attn = torch.nn.Parameter(torch.FloatTensor(2, ng_dim), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.ng_attn.data)
            self.l1_e0 = torch.nn.Linear(ng_dim, ng_dim//2)
            self.l1_e1 = torch.nn.Linear(ng_dim, ng_dim//2)

            self.l1_s = torch.nn.Linear(ng_score_dim, ng_score_dim)

            torch.nn.init.xavier_uniform_(self.l1_e0.weight.data)
            torch.nn.init.xavier_uniform_(self.l1_e1.weight.data)
            self.l1_e0.bias.data.zero_()
            self.l1_e1.bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.l1_s.weight.data)
            self.l1_s.bias.data.zero_()
            dim = arg_dim + ng_dim
            if self.use_ng_scores:
                dim += ng_score_dim
        else:
            dim = arg_dim
        logger.info('elmo_dim={}'.format(arg_dim))
        logger.info('ng_score_dim={}'.format(ng_score_dim))
        logger.info('ng_dim={}'.format(ng_dim))
        logger.info('ng_score_dim={}'.format(ng_score_dim))
        logger.info('dim={}'.format(dim))
        self.d1 = torch.nn.Dropout(p=self.dropout)

        self.l2= torch.nn.Linear(dim, dim//2)
        self.d2 = torch.nn.Dropout(p=self.dropout)

        self.l3= torch.nn.Linear(dim//2, dim//4)
        self.d3 = torch.nn.Dropout(p=self.dropout)

        self.l4= torch.nn.Linear(dim//4, n_rel_classes)

        torch.nn.init.xavier_uniform_(self.l1_0.weight.data)
        torch.nn.init.xavier_uniform_(self.l1_1.weight.data)
        torch.nn.init.xavier_uniform_(self.l2.weight.data)
        torch.nn.init.xavier_uniform_(self.l3.weight.data)
        torch.nn.init.xavier_uniform_(self.l4.weight.data)
        self.l1_0.bias.data.zero_()
        self.l1_1.bias.data.zero_()
        self.l2.bias.data.zero_()
        self.l3.bias.data.zero_()
        self.l4.bias.data.zero_()

        if use_class_weights:
            # hard coded
            self.class_weights = torch.FloatTensor([
                13.227513227513226,
                14.245014245014243,
                20.964360587002098,
                10.893246187363832,
                9.033423667570009,
                23.148148148148145,
                123.45679012345678,
                5.167958656330749
            ])
            self._loss = torch.nn.NLLLoss(weight=self.class_weights)
        else:
            self._loss = torch.nn.NLLLoss()

    @staticmethod
    def attend_context(x, attn):
        attention_score = torch.matmul(x, attn).squeeze()
        attention_score = F.softmax(attention_score, dim=1).view(x.size(0), x.size(1), 1)
        scored_x = x * attention_score
        x_context = torch.sum(scored_x, dim=1)
        return x_context

    def forward(self, x0, x1, x0_ng, x1_ng, x_ng):
        x0_context = self.attend_context(x0, self.arg_attn[0])
        x1_context = self.attend_context(x1, self.arg_attn[1])
        out1_0 = F.relu(self.l1_0(x0_context))
        out1_1 = F.relu(self.l1_1(x1_context))

        if self.use_ng:
            e0_context = self.attend_context(x0_ng, self.ng_attn[0])
            e1_context = self.attend_context(x1_ng, self.ng_attn[1])

            out1_e0 = F.relu(self.l1_e0(e0_context))
            out1_e1 = F.relu(self.l1_e1(e1_context))
            if self.use_ng_scores:
                x_ng = F.normalize(x_ng)
                out1_s = F.relu(self.l1_s(x_ng))
                out1 = torch.cat((out1_0, out1_1, out1_e0, out1_e1, out1_s), dim=1)
            else:
                out1 = torch.cat((out1_0, out1_1, out1_e0, out1_e1), dim=1)
        else:
            out1 = torch.cat((out1_0, out1_1), dim=1)

        out1 = self.d1(out1)
        out2 = self.d2(F.relu(self.l2(out1)))
        out3 = self.d3(F.relu(self.l3(out2)))
        out4 = F.log_softmax(self.l4(out3), dim=1)
        return out4

    def loss_func(self, probs, target):
        return self._loss(probs, target)

    def predict(self, x0, x1, x0_ng, x1_ng, x_ng):
        probs = self.forward(x0, x1, x0_ng, x1_ng, x_ng)
        _, pred = torch.max(probs, 1)
        return pred


def rel_output(rel, predicted_sense):
    new_rel = {}
    new_rel['DocID'] = rel['DocID']
    new_rel['ID'] = rel['ID']

    new_rel['Arg1'] = {}
    new_rel['Arg1']['TokenList'] = []
    for tok in rel['Arg1']['TokenList']:
        new_rel['Arg1']['TokenList'].append(tok[2])

    new_rel['Arg2'] = {}
    new_rel['Arg2']['TokenList'] = []
    for tok in rel['Arg2']['TokenList']:
        new_rel['Arg2']['TokenList'].append(tok[2])

    new_rel['Connective'] = {}
    new_rel['Connective']['TokenList'] = []
    for tok in rel['Connective']['TokenList']:
        new_rel['Connective']['TokenList'].append(tok[2])

    new_rel['Sense'] = [predicted_sense]
    new_rel['Type'] = rel['Type']
    return new_rel


def save(fld, model, optimizer, i_epoch, i_file, i_batch):
    fpath = os.path.join(ld, "model_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    torch.save(model.state_dict(), fpath)

    fpath = os.path.join(fld, "optim_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    torch.save(optimizer.state_dict(), fpath)

    fpath = os.path.join(fld, "argw_enc_{}_{}_{}.pt".format(i_epoch, i_file, i_batch))
    model.argw_encoder.save(fpath)


def save_losses(fld, losses):
    fpath = os.path.join(fld, 'losses.pkl')
    pkl.dump(losses, open(fpath, 'w'))


def save_scores(fld, scores, fname):
    fpath = os.path.join(fld, fname)
    pkl.dump(scores, open(fpath, 'w'))


def evaluate(model, dev_dataloader, dev_cm, dev_valid_senses):
    model.eval()
    all_ys, all_preds = [], []
    with torch.no_grad():
        for x, y in dev_dataloader:
            if args.use_ng:
                x0, x1, x0_ng, x1_ng, x_ng = x
                y = y.squeeze()
                if args.gpu_id is not None:
                    x0 = x0.cuda(args.gpu_id)
                    x1 = x1.cuda(args.gpu_id)
                    x0_ng = x0_ng.cuda(args.gpu_id)
                    x1_ng = x1_ng.cuda(args.gpu_id)
                    x_ng = x_ng.cuda(args.gpu_id)
            else:
                x0, x1 = x
                y = y.squeeze()
                if args.gpu_id is not None:
                    x0 = x0.cuda(args.gpu_id)
                    x1 = x1.cuda(args.gpu_id)
                    x0_ng, x1_ng, x_ng = None, None, None

            y_pred = model.predict(x0, x1, x0_ng, x1_ng, x_ng)
            all_ys.append(y)
            all_preds.append(y_pred.cpu())

    y_dev = torch.cat(all_ys, dim=0)
    y_dev_pred = torch.cat(all_preds, dim=0)

    dev_prec, dev_recall, dev_f1 = ds.scoring_cm(
        y_dev,
        y_dev_pred,
        dev_cm,
        dev_valid_senses, ds.CONLL16_IDX2REL
    )
    return dev_prec, dev_recall, dev_f1, y_dev, y_dev_pred


def train(model, optimizer, scheduler, train_dataset, dev_dataset, dev_cm, dev_valid_senses):
    # arg_lens = [config['arg0_max_len'], config['arg1_max_len']]
    dev_dataloader = DataLoader(dev_dataset,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            num_workers=1)
    train_dataloader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=1)

    losses = []
    dev_f1s = []
    count_patience = 0
    best_dev_f1, best_epoch, best_batch = -1, -1, -1
    logger.info("train_batch_size = {}".format(args.train_batch_size))
    logger.info("eval_batch_size = {}".format(args.eval_batch_size))
    for i_epoch in range(args.max_n_epoches):
        epoch_start = time.time()
        score_updated_epoch = False
        for i_batch, (x, y) in enumerate(train_dataloader):
            if y.shape[0] != args.train_batch_size:
                # skip the last batch
                continue
            y = y.squeeze()
            if args.use_ng:
                x0, x1, x0_ng, x1_ng, x_ng = x
                if args.gpu_id is not None:
                    x0 = x0.cuda(args.gpu_id)
                    x1 = x1.cuda(args.gpu_id)
                    x0_ng = x0_ng.cuda(args.gpu_id)
                    x1_ng = x1_ng.cuda(args.gpu_id)
                    x_ng = x_ng.cuda(args.gpu_id)
                    y = y.cuda(args.gpu_id)
            else:
                x0, x1 = x
                if args.gpu_id is not None:
                    x0 = x0.cuda(args.gpu_id)
                    x1 = x1.cuda(args.gpu_id)
                    y = y.squeeze().cuda(args.gpu_id)
                    x0_ng, x1_ng, x_ng = None, None, None
                    y = y.cuda(args.gpu_id)

            model.train()
            optimizer.zero_grad()
            out = model(x0, x1, x0_ng, x1_ng, x_ng)
            loss = model.loss_func(out, y)

            # step
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()

            losses.append(loss.item())

            # evaluate
            dev_prec, dev_recall, dev_f1, _, _ = evaluate(
                model,
                dev_dataloader,
                dev_cm,
                dev_valid_senses
            )
            dev_f1s.append(dev_f1)

            logger.info("{}, {}: loss={}, time={}".format(
                i_epoch, i_batch, loss.item(), time.time()-epoch_start))
            # logger.info("dev: prec={}, recall={}, f1={}".format(
            #     dev_prec, dev_recall, dev_f1))
            if dev_f1 > best_dev_f1:
                logger.info("best dev: prec={}, recall={}, f1={}".format(
                    dev_prec, dev_recall, dev_f1))
                best_dev_f1 = dev_f1
                best_epoch = i_epoch
                best_batch = i_batch
                fpath = os.path.join(args.output_folder, 'best_model.pt')
                torch.save(model.state_dict(), fpath)
                score_updated_epoch = True

        logger.info('done epoch: {}s'.format(time.time()-epoch_start))
        if score_updated_epoch:
            count_patience = 0
        else:
            count_patience += 1
            # stop by patience
            if args.patience is not None and count_patience >= args.patience:
                logger.info('count_patience = {}. stop training.'.format(count_patience))
                break

    logger.info("{}-{}: best dev f1 = {}".format(best_epoch, best_batch, best_dev_f1))
    fpath = os.path.join(args.output_folder, "losses.pkl")
    pkl.dump(losses, open(fpath, 'wb'))
    fpath = os.path.join(args.output_folder, "dev_f1s.pkl")
    pkl.dump(dev_f1s, open(fpath, 'wb'))


def main():
    set_seed(args.gpu_id, args.seed)

    if args.mode == 'train':
        train_dataset = CONLL16Dataset(args.train_dir, use_ng_features=args.use_ng)

        dev_rels = [json.loads(line) for line in open(args.dev_rel)]
        dev_rels = [rel for rel in dev_rels if rel['Type'] != 'Explicit' and rel['Sense'][0] in ds.CONLL16_REL2IDX]
        dev_cm, dev_valid_senses = ds.create_cm(dev_rels, ds.CONLL16_REL2IDX)

        dev_dataset = CONLL16Dataset(args.dev_dir, use_ng_features=args.use_ng)

        model = AttentionNN(len(ds.CONLL16_REL2IDX), dropout=args.dropout, use_ng=args.use_ng, use_ng_scores=args.use_ng_scores,
                            use_class_weights=args.use_class_weights)
        if args.gpu_id is not None:
            model = model.cuda(args.gpu_id)

        if args.optim == 'adamw':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                                     "weight_decay": args.weight_decay},
                                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}

            ]
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

            total_steps = len(train_dataset) * args.max_n_epoches
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=total_steps)
        elif args.optim == 'adagrad':
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
            scheduler = None
        logger.info('optimizer: {}'.format(args.optim))
        logger.info("initial learning rate = {}".format(args.learning_rate))
        logger.info("dropout rate = {}".format(args.dropout))

        train(model, optimizer, scheduler, train_dataset, dev_dataset, dev_cm, dev_valid_senses)

    else: # test
        output_rels = [json.loads(line) for line in open(args.test_rel)]
        test_rels = [rel for rel in output_rels if rel['Type'] != 'Explicit'
                     and rel['Sense'][0] in ds.CONLL16_REL2IDX] # include Implicit, EntRel, AltLex

        count_senses = {}
        count_types = {}
        for rel in test_rels:
            s = rel['Sense'][0]
            if s not in count_senses:
                count_senses[s] = 0
            count_senses[s] += 1

            t = rel['Type']
            if t not in count_types:
                count_types[t] = 0
            count_types[t] += 1
        logger.info('count_senses: {}'.format(count_senses))
        logger.info('count_types: {}'.format(count_types))

        test_cm, test_valid_senses = ds.create_cm(test_rels, ds.CONLL16_REL2IDX)
        test_dataset = CONLL16Dataset(args.test_dir, use_ng_features=args.use_ng)

        model = AttentionNN(len(ds.CONLL16_REL2IDX), dropout=args.dropout, use_ng=args.use_ng, use_ng_scores=args.use_ng_scores,
                            use_class_weights=args.use_class_weights)
        logger.info('load state_dict from {}'.format(args.model_file))
        model.load_state_dict(
            torch.load(args.model_file, map_location=lambda storage, location: storage)
        )
        if args.gpu_id is not None:
            model = model.cuda(args.gpu_id)

        test_dataloader = DataLoader(test_dataset,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                num_workers=1)

        # predict
        test_prec, test_recall, test_f1, y, y_pred = evaluate(
            model,
            test_dataloader,
            test_cm,
            test_valid_senses
        )
        logger.info('prec={}, recall={}, f1={}'.format(
            test_prec, test_recall, test_f1))

        # dump
        fw_path = os.path.join(args.output_folder, 'predicted_relations.json')
        logger.info('saving {}...'.format(fw_path))
        fw = open(fw_path, 'w')

        # write the predicted implicit rels
        predicted_rel_ids = set()
        for i, rel in enumerate(test_rels):
            final_pred = y_pred[i].item()
            new_rel = rel_output(rel, ds.CONLL16_IDX2REL[final_pred])
            predicted_rel_ids.add(new_rel['ID'])
            fw.write(json.dumps(new_rel) + '\n')

        # for explicit rels we simply predict the majority class, EntRel
        for i, rel in enumerate(output_rels):
            if rel['ID'] not in predicted_rel_ids:
                new_rel = rel_output(rel, "EntRel")
                fw.write(json.dumps(new_rel) + '\n')

        fw.close()


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
