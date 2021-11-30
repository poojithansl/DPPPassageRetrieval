#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Most of the tokenization code here is copied from Facebook/DPR & DrQA codebase to avoid adding an extra dependency
"""

import argparse
import copy
import json
import logging
import re
import unicodedata
from tqdm import tqdm
import numpy as np

import regex
from dpp import DPP
DEBUG = 1

logger = logging.getLogger(__name__)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def which_answer(text, answers, tokenizer, regex=False):
    text = _normalize(text)
    if regex:
        for ix, ans in enumerate(answers):
            ans = _normalize(ans)
            # breakpoint()
            if regex_match(text, ans):
                return ix
    else:
        text = tokenizer.tokenize(text).words(uncased=True)
        for ix, ans in enumerate(answers):
            ans = _normalize(ans)
            ans = tokenizer.tokenize(ans).words(uncased=True)
            for i in range(0, len(text) - len(ans) + 1):
                if ans == text[i: i + len(ans)]:
                    return ix
    return -1


def evaluate_retrieval(input, batch_size, topk, multi_answer=True, regex=False):
    dpp = DPP(input, batch_size)
    dataReader = dpp.Lagrangian.dataReader

    tokenizer = SimpleTokenizer()
    data_dict = dataReader.get_data_dict()

    accuracy = { topk : [] }
    secondary_accuracy = { topk : [] }
    count = 0
    for batchPassages in dpp.runDPP(topk):
        for ix_passage_list in batchPassages: # 1*5
            answers = data_dict[str(count)]['answers']
            contexts = data_dict[str(count)]['contexts']
            count += 1
            # Initializing answer checks
            present_answers = [0] * len(answers)
            has_ans = False
            for idx in ix_passage_list:
                # print(idx)
                ctx = contexts[idx]
                if multi_answer:
                    text = ctx['text'].split('\n')[1]  # [0] is title, [1] is text
                    present_answer_id = which_answer(text, answers, tokenizer, regex)
                    if present_answer_id != -1:
                        present_answers[present_answer_id] = 1
                else:
                    if 'has_answer' in ctx:
                        if ctx['has_answer']:
                            has_ans= True
                            break
                    if has_ans:
                        break
            if multi_answer:
                # if n <= k, all n should be in topK
                # else if n > k, all k passages should be having an answer
                if len(answers) <= topk:
                    acc = all(present_answers)
                else:
                    acc = present_answers.count(1) == topk
                if len(answers) > 1:
                    secondary_accuracy[topk].append(acc)
            else:
                # For single answer just any of the answers need to be present
                acc = has_ans==True
            accuracy[topk].append(acc)
        if DEBUG:
            print(np.sum(accuracy[topk]), count)
    print(f'Top{topk}\taccuracy: {np.mean(accuracy[topk])}')
    print(f'Number of samples right: {np.sum(accuracy[topk])}')
    print(f'Number of samples: {len(accuracy[topk])}')
    if len(secondary_accuracy[topk]) >= 1:
        print(f'Secondary Top{topk}\taccuracy: {np.mean(secondary_accuracy[topk])}')
        print(f'Secondary samples correct: {np.sum(secondary_accuracy[topk])}')
        print(f'Number of secondary samples: {len(secondary_accuracy[topk])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required = False, default = '../data/nq-test.json', help = 'Input json file')
    parser.add_argument('--bs', type=int, required = False, default = 8, help = 'batch size')
    parser.add_argument('--topk', type=int, default = 5, nargs='+', help="topk to evaluate")
    parser.add_argument('--mans', action='store_true', default=True, help="multiple answers accuracy")
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    args = parser.parse_args()

    evaluate_retrieval(args.input, args.bs, args.topk, args.mans, args.regex)
