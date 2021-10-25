import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class CrossLayer(Module):
    """
    MultiLayer
    """
    def __init__(self, L1_dim, L2_dim, bias=True, bet_weight=True):
        super(CrossLayer, self).__init__()
        self.L1_dim = L1_dim
        self.L2_dim = L2_dim
        self.bet_weight = bet_weight
        self.weight = Parameter(torch.FloatTensor(L1_dim, L2_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(L2_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, L1_features, L2_features):
        if self.bet_weight:
            temp = torch.mm(L1_features, self.weight)
            output = torch.mm(temp, torch.t(L2_features))
            if self.bias is not None:
                output = output + self.bias
        else:
            output = torch.mm(L1_features, torch.t(L2_features))
        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.L1_dim) + ' -> ' \
               + str(self.L2_dim) + ')'