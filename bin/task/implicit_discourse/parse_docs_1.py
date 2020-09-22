import os
import sys
import json
import time
import random
import pickle as pkl
import argparse
import logging.config
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict
from itertools import combinations

from tqdm import tqdm
from stanza.server import CoreNLPClient

from nglib.common import utils


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Parse CONLL16 with Stanford CoreNLP')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='input config')
    parser.add_argument('split', metavar='SPLIT', choices=['train', 'dev', 'test', 'blind_test'],
                        help='split.')
    parser.add_argument('output_file', metavar='OUTPUT_FILE',
                        help='output file.')

    parser.add_argument('-p', '--nlp_server_port', type=int, default=9001,
                        help='Server port for Stanford CoreNLP')
    parser.add_argument('--seed', type=int, default=135,
                        help='seed for random')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def main():
    assert config['config_target'] == 'conll16_discourse'

    # start CoreNLP server manually
    # java -Xmx16G -cp "/homes/lee2226/scratch2/stanford-corenlp-full-2020-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 60000 -threads 5 -maxCharLength 100000 -preload tokenize,ssplit, pos,lemma,ner, parse,depparse,coref,kbp -outputFormat json

    # use use the tokenization from the given parse file and re-parse them with our CoreNLP
    split_dir = config['{}_dir'.format(args.split)]
    parse_fpath = os.path.join(split_dir, 'parses.json')
    logger.info('loading {}...'.format(parse_fpath))
    old_parses = json.load(open(parse_fpath, 'r'))

    fw = open(args.output_file, 'w')
    properties = {
            'tokenize.whitespace': True,
            'tokenize.keepeol': True,
            'ssplit.eolonly': True,
            'ner.useSUTime': False
            }
    annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
    with CoreNLPClient(
        annotators=annotators,
        properties=properties,
        timeout=300000,
        endpoint='http://localhost:{}'.format(args.nlp_server_port),
        start_server=False
    ) as client:
        for doc_id, parse in tqdm(old_parses.items()):
            sents = []
            for i_sent, sent in enumerate(parse['sentences']):
                words = [w[0] for w in sent['words']]
                sent_text = ' '.join(words)
                sents.append(sent_text)
            all_text = '\n'.join(sents)


            try:
                ann = client.annotate(all_text,
                                      annotators=annotators,
                                      properties=properties,
                                      output_format='json')
                ann['doc_id'] = doc_id

                # verify lengths
                assert len(ann['sentences']) == len(parse['sentences']), 'ssplit mismatch'
                for i_sent in range(len(parse['sentences'])):
                    n_words = len(parse['sentences'][i_sent]['words'])
                    assert len(ann['sentences'][i_sent]['tokens']) == n_words

                out = json.dumps(ann)
                fw.write(out + '\n')
            except:
                logger.warning('failed parsing {}'.format(doc_id))
    fw.close()


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    random.seed(args.seed)
    config = json.load(open(args.config_file))
    main()
