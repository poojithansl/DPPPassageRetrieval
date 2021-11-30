
from sentence_transformers import SentenceTransformer, util
from read_data import Data
import torch
import argparse

DEBUG = 0

class LagrangianMatrix:
    def __init__(self, batch_size, input):
        self.Sim_model = SentenceTransformer('sentence-transformers/stsb-mpnet-base-v2')
        self.Qty_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        self.dataReader = Data(batch_size, input)
        self.batch_size = batch_size

    def compute_matrix(self):
        # passages is batch_size * 100
        for query_list, passages_list in self.dataReader.load():
            if DEBUG:
                print(len(passages_list)) # batch size
                print(len(passages_list[0])) # 100
            ltensorList = []
            for ix in range(len(passages_list)):
                passages = passages_list[ix]
                query = query_list[ix]
                #Quality
                qty_query_encoding = self.Qty_model.encode(query, convert_to_tensor=True)
                qty_passage_encoding = self.Qty_model.encode(passages, convert_to_tensor=True)
                scores = util.dot_score(qty_query_encoding, qty_passage_encoding)
                qualityMatrix = torch.mul(scores, scores.T)
                # Similarity
                sim_passage_encoding = self.Sim_model.encode(passages, convert_to_tensor=True)
                simMatrix = util.pytorch_cos_sim(sim_passage_encoding, sim_passage_encoding)
                # L matrix
                lmatrix = torch.mul(qualityMatrix, simMatrix)
                ltensorList.append(lmatrix)
                if DEBUG:
                    if torch.isnan(torch.min(lmatrix)):
                        breakpoint()

            #simTensor is batch_size * 100 * 100
            if DEBUG:
                print(qualityMatrix.size())
                print(lmatrix.size())
                print(ltensor.size())
                print(scores.size())
                print(simMatrix.size())
                print(sim_passage_encoding.size())
            ltensor = torch.stack(ltensorList)
            yield(ltensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read top 100 passages ')
    parser.add_argument('--input', required = False, default = '../data/nq-test.json', help = 'Input json file')
    parser.add_argument('--bs', required = False, default = 8, help = 'batch size')
    args = parser.parse_args()

    Lmatrix = LagrangianMatrix(int(args.bs), args.input)
    Lmatrix.compute_matrix()
