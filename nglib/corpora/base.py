import re
import logging
from lxml import etree

import spacy


logger = logging.getLogger(__name__)


class CorpusBase:
    def __init__(self):
        pass


class TimeBankBase(CorpusBase):
    def __init__(self):
        super(TimeBankBase, self).__init__()
        self.nlp = None

    def _get_sent_tok_idx(self, sents, tok_start, tok_end):
        for i_sent, sent in enumerate(sents):
            if tok_start >= sent['tok_start_idx'] and tok_end <= sent['tok_end_idx']:
                sent_tok_start = tok_start - sent['tok_start_idx']
                sent_tok_end = sent_tok_start + (tok_end - tok_start)
                return i_sent, sent_tok_start, sent_tok_end
        # this usually means that we have ssplit erros.
        # logger.debug(
        #     'Unable to find sent_idx in default mode. We align to the sentence end')
        # for i_sent, sent in enumerate(sents):
        #     if tok_start >= sent['tok_start_idx']:
        #         sent_tok_start = tok_start - sent['tok_start_idx']
        #         sent_tok_end = sent['tok_end_idx']
        #         return i_sent, sent_tok_start, sent_tok_end
        raise ValueError('sentence split error')

    def _get_sentences(self, tokens, events, timexs, doc_id, manual_corrections=False):
        old_tokens = []
        for tok in tokens:
            if (tok.startswith('\'s')
                    or tok.startswith('\'ve')
                    or tok.startswith('\'re')
                    or tok.startswith('n\'t')):
                old_tokens[-1] += tok
            else:
                old_tokens.append(tok)
        text = ' '.join(old_tokens)

        # this is mainly for sentence split (document-level)
        if self.nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
        doc = self.nlp(text)

        # get map from old to new tokens
        new_tokens = [tok.text for tok in doc]
        old2new = {}
        i, j = 0, 0
        while True:
            if tokens[i] == new_tokens[j]:
                old2new[(i,)] = (j, )
                i += 1
                j += 1
            else:
                if len(tokens[i]) > len(new_tokens[j]):
                    new_comb = new_tokens[j] + new_tokens[j+1]
                    if tokens[i] == new_comb:
                        old2new[(i, )] = (j, j+1)
                        i += 1
                        j += 2
                    else:
                        raise ValueError('parsing error')
                else:
                    old_comb = tokens[i] + tokens[i+1]
                    if old_comb == new_tokens[j]:
                        old2new[(i, i+1)] = (j, )
                        i += 2
                        j += 1
                    else:
                        raise ValueError('parsing error')

            if i >= len(doc) or j >= len(doc):
                old2new[(i, )] = (j, )
                break

        sents = []
        for i_sent, sent in enumerate(doc.sents):
            s = {
                    'sent_idx': i_sent,
                    'tok_start_idx': sent.start,
                    'tok_end_idx': sent.end
                    }
            sents.append(s)

        # some manual corrections
        if manual_corrections:
            if doc_id == 'APW19980213.1380':
                # merge sent 3, 4
                new_sents = []
                for i in range(len(sents)):
                    if i < 3:
                        new_sents.append(sents[i])
                    elif i == 3:
                        sent = {
                            'sent_idx': i,
                            'tok_start_idx': sents[i]['tok_start_idx'],
                            'tok_end_idx': sents[i+1]['tok_end_idx']
                        }
                        new_sents.append(sent)
                    elif i == 4:
                        continue
                    else:
                        sent = sents[i]
                        sent['sent_idx'] = i-1
                        new_sents.append(sent)
                sents = new_sents

        def __update(event):
            if 'tok_start_idx' in event:
                start_idx = event['tok_start_idx']
                new_start_idx = old2new[(start_idx, )][0]
                end_idx = event['tok_end_idx']
                new_end_idx = old2new[(end_idx, )][-1]
                # if ' '.join(new_tokens[new_start_idx:new_end_idx]) != event['text']:
                    # logger.warning('event text mismatch: {}'.format(event))
                event['tok_start_idx'] = new_start_idx
                event['tok_end_idx'] = new_end_idx

                sent_idx, sent_tok_start_idx, sent_tok_end_idx = \
                        self._get_sent_tok_idx(sents, new_start_idx, new_end_idx)

                event['sent_idx'] = sent_idx
                event['sent_tok_start_idx'] = sent_tok_start_idx
                event['sent_tok_end_idx'] = sent_tok_end_idx
                # if ' '.join(new_tokens[sents[sent_idx]['tok_start_idx']: sents[sent_idx]['tok_end_idx']][sent_tok_start_idx: sent_tok_end_idx]) != event['text']:
                    # logger.warning('event text mismatch: {}'.format(event))
            else:
                logger.debug('{}-{} has no tok_start_idx'.format(doc_id, event))
        # update events
        for eid, e in events.items():
            __update(e)

        # update timexs
        for tid, timex in timexs.items():
            __update(timex)

        return sents, new_tokens, events, timexs

    def _get_doc_creation_time(self, root):
        dct = root.findall('DCT')
        assert len(dct) == 1
        dct = dct[0]
        dct_timex = root[1].getchildren()[0]
        dct = {
                'tid': dct_timex.get('tid'),
                'type': dct_timex.get('type'),
                'value': dct_timex.get('value'),
                'text': dct_timex.text
                }
        assert dct['tid'] == 't0'
        return dct

    def _get_tlinks(self, root, eiid2eid, no_self_loop=True):
        tlinks = {}
        for elem in root.findall('TLINK'):
            eiid1 = elem.get('eventInstanceID')
            t1 = elem.get('timeID')
            eiid2 = elem.get('relatedToEventInstance')
            t2 = elem.get('relatedToTime')
            rtype = elem.get('relType')
            lid = elem.get('lid')

            eid1 = eiid2eid[eiid1] if eiid1 else None
            eid2 = eiid2eid[eiid2] if eiid2 else None

            if eid1 and eid2:
                edge_type = 'ee'
                source, target = eid1, eid2
            elif eid1 and t2:
                edge_type = 'et'
                source, target = eid1, t2
            elif t1 and eid2:
                edge_type = 'te'
                source, target = t1, eid2
            elif t1 and t2:
                edge_type = 'tt'
                source, target = t1, t2
            else:
                raise ValueError('unsupported edge type.')

            if no_self_loop and source == target:
                logger.debug('skip self-loop: lid={}: {} - {}'.format(
                    lid, source, target))
                continue

            tlink = {
                    'source': source,
                    'target': target,
                    'edge_type': edge_type,
                    'rel_type': rtype.lower(),
                    'link_id': lid
                    }
            tlinks[lid] = tlink
        return tlinks

    def _get_tokens(self, root, events, dct, include_dct=True):
        text_root = root.findall('TEXT')
        assert len(text_root) == 1
        text_root = text_root[0]

        if text_root.text is None:
            first_span = text_root.getchildren()[0].text
        else:
            first_span = text_root.text
        assert len(first_span) > 0

        texts = text_root.xpath('//text()')
        children = text_root.getchildren()
        start = texts.index(first_span)

        tokens, timexs = [], {}
        cur_child_idx = 0
        for span in texts[start:]:
            s = str(span)
            s = s.strip()
            stoks = self._tokenize(s)
            if cur_child_idx < len(children) and s == children[cur_child_idx].text.strip():
                start_idx = len(tokens)
                end_idx = start_idx + len(stoks)
                if children[cur_child_idx].tag == 'EVENT':
                    eid = children[cur_child_idx].get('eid')
                    ecls = children[cur_child_idx].get('class')
                    if eid in events:
                        events[eid]['class'] = ecls
                        events[eid]['tok_start_idx'] = start_idx
                        events[eid]['tok_end_idx'] = end_idx
                        events[eid]['text'] = s
                    else:
                        logger.debug('eid {} not in events dict'.format(eid))
                elif children[cur_child_idx].tag == 'TIMEX3':
                    tid = children[cur_child_idx].get('tid')
                    ttype = children[cur_child_idx].get('type')
                    tvalue = children[cur_child_idx].get('value')
                    timexs[tid] = {
                            'tid': tid,
                            'type': ttype,
                            'value': tvalue,
                            'tok_start_idx': start_idx,
                            'tok_end_idx': end_idx,
                            'text': s
                            }
                else:
                    # APW19980813.1117 has SIGNAL
                    logger.debug('unknown event tag: {}'.format(
                        children[cur_child_idx].tag))

                cur_child_idx += 1
            tokens += stoks

        if include_dct:
            assert dct['tid'] not in timexs
            timexs[dct['tid']] = {
                'tid': dct['tid'],
                'type': dct['type'],
                'value': dct['value'],
                # 'tok_start_idx': start_idx,
                # 'tok_end_idx': end_idx,
                'text': dct['text']
            }
        return tokens, timexs, events

    def _get_events(self, root):
        eiid2eid = {}
        events = {}
        for elem in root.findall('MAKEINSTANCE'):
            if elem.get('signalID') is not None:
                logger.debug('has SIGNAL instance.')
            aspect = elem.get('aspect')
            eiid = elem.get('eiid')
            eid = elem.get('eventID')
            polarity = elem.get('polarity')
            tense = elem.get('tense')
            e = {
                    'aspect': aspect,
                    'eiid': eiid,
                    'eid': eid,
                    'polarity': polarity,
                    'tense': tense
                    }
            eiid2eid[eiid] = eid
            if eid in events:
                logger.warning('eid to multiple eiids: {} -> {}'.format(
                    eid, [eiid, events[eid]['eiid']]))
                assert e['eid'] == events[eid]['eid']
            else:
                events[eid] = e
        return events, eiid2eid

    def _tokenize(self, s):
        if self.nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
        s = re.sub(r'\n', ' ', s)
        doc = self.nlp(s)
        return [tok.text for tok in doc if tok.text.strip()]
