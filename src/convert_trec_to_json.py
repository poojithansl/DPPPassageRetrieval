import argparse
import json
import os
from tqdm import tqdm

from pyserini.search import SimpleSearcher, get_topics
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an TREC run to DPR retrieval result json.')
    parser.add_argument('--topics', required=True, help='topic name')
    parser.add_argument('--index', required=True, help='Anserini Index that contains raw')
    parser.add_argument('--input', required=True, help='Input TREC run file.')
    parser.add_argument('--store-raw', action='store_true', help='Store raw text of passage')
    parser.add_argument('--regex', action='store_true', default=False, help="regex match")
    parser.add_argument('--output', required=True, help='Output DPR Retrieval json file.')
    parser.add_argument('--top', required=True, help='Get top k')
    args = parser.parse_args()

    qas = get_topics(args.topics)

    if os.path.exists(args.index):
        searcher = SimpleSearcher(args.index)
    else:
        searcher = SimpleSearcher.from_prebuilt_index(args.index)
    if not searcher:
        exit()

    retrieval = {}
    tokenizer = SimpleTokenizer()
    with open(args.input) as f_in:
        for line in tqdm(f_in.readlines()):
            question_id, _, doc_id, rank, score, _ = line.strip().split()
            if rank > 100:
              continue
            question_id = int(question_id)
            question = qas[question_id]['title']
            answers = qas[question_id]['answers']
            if answers[0] == '"':
                answers = answers[1:-1].replace('""', '"')
            answers = eval(answers)
            ctx = json.loads(searcher.doc(doc_id).raw())['contents']
            if question_id not in retrieval:
                retrieval[question_id] = {'question': question, 'answers': answers, 'contexts': []}
            title, text = ctx.split('\n')
            answer_exist = has_answers(text, answers, tokenizer, args.regex)
            if args.store_raw:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id,
                     'score': score,
                     'text': ctx,
                     'has_answer': answer_exist}
                )
            else:
                retrieval[question_id]['contexts'].append(
                    {'docid': doc_id, 'score': score, 'has_answer': answer_exist}
                )

    json.dump(retrieval, open(args.output, 'w'), indent=4)
