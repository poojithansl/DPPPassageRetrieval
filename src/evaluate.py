from dpp import DPP
import argparse

class Evaluate:
    def __init__(self, input, batch_size):
        self.dpp = DPP(input, batch_size)
        self.dataReader = self.dpp.Lagrangian.dataReader
        pass

    def calc_single_recall(self, topk):
        count = 0
        countCorrectRetrieval = 0
        # totalNumQueries = 0
        self.data_dict = self.dataReader.get_data_dict()
        for batchPassages in self.dpp.runDPP(topk):
            for passage_list in batchPassages: # 1*5
                query = self.data_dict[str(count)]["question"]
                # print(query)
                contexts = self.data_dict[str(count)]["contexts"]
                has_answer = False
                for ix in passage_list:
                    picked_passage = contexts[ix]["text"]
                    # print(picked_passage)
                    has_answer = max(contexts[ix]["has_answer"], has_answer)
                countCorrectRetrieval += has_answer==True
                count += 1
            print(countCorrectRetrieval, count)
        print(countCorrectRetrieval, count)


def run(input, batch_size, topk):
    eval = Evaluate(input, batch_size)
    eval.calc_single_recall(topk)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--input', required = False, default = '../data/nq-test.json', help = 'Input json file')
    parser.add_argument('--bs', required = False, default = 8, help = 'batch size')
    parser.add_argument('--k', required = False, default = 5, help = 'top most passages')
    args = parser.parse_args()

    run(args.input, int(args.bs), int(args.k))
