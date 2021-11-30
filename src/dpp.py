from calc_matrix_l import LagrangianMatrix
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
DEBUG = 0

class DPP:
    def __init__(self, input, batch_size):
        self.Lagrangian = LagrangianMatrix(batch_size, input)
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        pass

    def runDPP(self, max_samples):
        for matrix_l in self.Lagrangian.compute_matrix():
            result = self.compute(matrix_l, max_samples)
            yield(result)

    def compute(self, matrix_l, max_samples):
        if DEBUG:
            print(matrix_l.shape)
        with torch.no_grad():
            count = 1 # counts of chosen elements
            N = matrix_l.size()[1] # counts of elements
            B = matrix_l.size()[0] # batch size
            C = torch.zeros(B, N, 1).to(self.device)  # extendable vector set

            index = torch.LongTensor().to(self.device)
            for _ in range(B):
                index = torch.cat([index, torch.linspace(0, N - 1, N).long().to(self.device)])
            index = index.view(B, 1, -1)  # index for select diagnol elements

            D = matrix_l.gather(1, index).view(B, -1)  # diagnol elements(quality) set
            mask = torch.ones(B, N).to(self.device) # chosen elements mask tensor which represent chosen elements as 0
            J = torch.argmax(torch.log(D * mask), dim=1)  # current chosen elements set
            mask = mask.scatter_(1, J.view(-1, 1), 0.0) # mask chosen elements

            increment = torch.LongTensor([]).to(self.device)
            for idx in range(B):
                increment = torch.cat([increment, torch.LongTensor([idx * N]).to(self.device)]) # for transforming index batchwise to non batchwise

            while count < max_samples:
                candidate = torch.nonzero(mask)[:, 1].view(B, -1)  # unchosen elements
                c_extend = torch.zeros(B, N, 1).to(self.device) # changements record tensor for c
                d_minus = torch.zeros(B, N).to(self.device) # changements record tensor for d

                for idx in range(N - count): # iterate all cadidate elements
                    i = candidate[:, idx] + increment
                    j = J + increment # get one candidate index batchwise

                    temp = matrix_l.contiguous().view(B * N, -1)
                    temp = torch.index_select(temp, 0, j)
                    matrix_select = torch.index_select(temp.view(1, -1), 1, i).view(-1) # make shape transformation twice to select tensor items based on index matrix

                    c_select_i = torch.index_select(C.contiguous().view(B * N,-1), 0, i).view(B, -1, 1)
                    c_select_j = torch.index_select(C.contiguous().view(B * N,-1), 0, j).view(B, -1, 1) # make shape transformation once to select c elements(vector) based on index

                    c_dot = torch.bmm(c_select_i.permute(0,2,1), c_select_j).view(-1) # compute tensor(vector) inner product

                    d_select_j =D.gather(1, J.view(B,-1)).view(-1) # select elements in d based on index j

                    e = (matrix_select - c_dot) / d_select_j # compute e according to formula

                    c_extend.scatter_(1, (i - increment).view(B, 1, 1), e.view(B, 1, 1)) # store extended part for c
                    d_minus.scatter_(1, (i - increment).view(B, 1), (e*e).view(B, 1)) # store minus part for d

                C = torch.cat([C, c_extend], dim=2)
                D = D - d_minus # apply changements after iteration
                J = torch.argmax(torch.log(D * mask), dim=1)  # select max elements based on modified D excluding chosen elements
                mask = mask.scatter_(1, J.view(-1, 1), 0.0) # mask chosen elements
                if DEBUG:
                    print(mask.shape)
                    print("Count:" + str(count))
                count += 1

            res = torch.nonzero(1 - mask)[:, 1].view(B, -1)
            if DEBUG:
                print(res.shape)
            if list(res.shape) != [B, max_samples]:
                nanpresent = False
                if torch.isnan(torch.min(matrix_l)):
                    nanpresent = True
                breakpoint()
            return res


def run(input, batch_size, topk):
    dpp = DPP(input, batch_size)
    dpp.runDPP(topk)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read top 100 passages ')
    parser.add_argument('--input', required = False, default = '../data/nq-test.json', help = 'Input json file')
    parser.add_argument('--bs', required = False, default = 8, help = 'batch size')
    parser.add_argument('--k', required = False, default = 5, help = 'top most passages')
    args = parser.parse_args()

    run(args.input, int(args.bs), int(args.k))
