import json
import argparse
import re
import os
DEBUG = 0

class Data:
    def __init__(self, batch_size, file):
        self.batch_size = batch_size
        self.file = file

    def clean(self, s):
        res = re.sub(r'[^\w\s]', '', s)
        return res

    def load(self):
        with open(self.file, 'r') as fp:
            self.data_dict = json.load(fp)
        if DEBUG:
            print(self.data_dict.keys())
        questions_len = len(self.data_dict.keys())
        i = 0
        while i < questions_len:
            questions_list = []
            passages_list = []
            for j in range(self.batch_size):
                questions_list.append(self.clean(self.data_dict[str(i)]['question']))
                contexts_list = self.data_dict[str(i)]['contexts']
                qpass_list = []
                for passage_item in contexts_list:
                    qpass_list.append(self.clean(passage_item['text'].strip()))
                passages_list.append(qpass_list)
                if DEBUG:
                    print(qpass_list[-1])
                    print(questions_list[-1])
                i += 1
                if i >= questions_len:
                    break
            yield(questions_list, passages_list)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Read top 100 passages ')
    parser.add_argument('--input', required = False, default = '../data/nq-test.json', help = 'Input json file')
    args = parser.parse_args()

    dataItem = Data(8, args.input)
    dataItem.load()
