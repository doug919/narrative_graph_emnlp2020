import os
import json
import logging

from .base import CorpusBase


class GigawordNYT(CorpusBase):
    def __init__(self, dir_path):
        '''process Gigaword NYT corpus in CoreNLP parsed json format
        '''
        super(GigawordNYT, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.train_fdir = os.path.join(dir_path, 'train')
        self.dev_fdir = os.path.join(dir_path, 'dev')
        self.test_fdir = os.path.join(dir_path, 'test')

    def load_parses(self, data_dir):
        parses = {}
        fs = [d for d in os.listdir(data_dir) if d.endswith('.json')]
        for f in fs:
            fpath = os.path.join(data_dir, f)
            self.logger.info('loading {}...'.format(fpath))
            with open(fpath, 'r') as fr:
                for line in fr:
                    p = json.loads(line)
                    parses[p['doc_id']] = p
        return parses

    def _iterate_docs(self, split):
        if split == 'train':
            fdir = self.train_fdir
        elif split == 'dev':
            fdir = self.dev_fdir
        elif split == 'test':
            fdir = self.test_fdir
        else:
            raise ValueError("split type {} is not supported")

        fs = sorted([d for d in os.listdir(fdir) if d.endswith('.json')])
        for f in fs:
            fpath = os.path.join(fdir, f)
            self.logger.info('loading {}...'.format(fpath))
            with open(fpath, 'r') as fr:
                for line in fr:
                    p = json.loads(line)
                    yield p
