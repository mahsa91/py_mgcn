import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, CrossLayer
from utils import between_index


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.nclass = nclass

        layers = []
        layers.append(GraphConvolution(nfeat, nhid[0]))
        for i in range(len(nhid)-1):
            layers.append(GraphConvolution(nhid[i], nhid[i+1]))
        if nclass > 1:
            layers.append(GraphConvolution(nhid[-1], nclass))
        self.gc = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x, adj):
        end_layer = len(self.gc)-1 if self.nclass > 1 else len(self.gc)
        for i in range(end_layer):
            x = F.relu(self.gc[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        if self.nclass > 1:
            classifier = self.gc[-1](x, adj)
            return F.log_softmax(classifier, dim=1), x
        else:
            return None, x


class CCN(nn.Module):
    def __init__(self, n_inputs, inputs_nfeat, inputs_nhid, inputs_nclass, dropout):
        super(CCN, self).__init__()

        # Features of network
        self.n_inputs = n_inputs
        self.inputs_nfeat = inputs_nfeat
        self.inputs_nhid = inputs_nhid
        self.inputs_nclass = inputs_nclass
        self.dropout = dropout

        # Every single layer
        temp_in_layer = []
        temp_bet_layer = []
        for i in range(n_inputs):
            temp_in_layer.append(GCN(inputs_nfeat[i], inputs_nhid[i], inputs_nclass[i], dropout))
        self.in_layer = nn.ModuleList(temp_in_layer)

        # Between layers
        for i in range(n_inputs):
            for j in range(i+1,n_inputs):
                temp_bet_layer.append(CrossLayer(inputs_nhid[i][-1], inputs_nhid[j][-1], bias=True, bet_weight=False))
        self.bet_layer = nn.ModuleList(temp_bet_layer)

    def forward(self, xs, adjs):
        classes_labels = []
        bet_layers_output = []
        in_layers_output = []
        features = []
        # Single layer forward to find features and classes
        for i in range(self.n_inputs):
            class_labels, feature = self.in_layer[i](xs[i], adjs[i])
            classes_labels.append(class_labels)
            features.append(feature)

            temp = torch.mm(feature, torch.t(feature))
            in_layers_output.append(temp)

        # Find between layers with CCN
        for i in range(self.n_inputs):
            for j in range(i+1,self.n_inputs):
                temp = self.bet_layer[between_index(self.n_inputs, i, j)](features[i], features[j])
                bet_layers_output.append(temp)

        return classes_labels, bet_layers_output, in_layers_output


class AggCCN(nn.Module):
    def __init__(self, n_inputs, inputs_nfeat, inputs_nhid, inputs_nclass, dropout):
        super(AggCCN, self).__init__()

        # Features of network
        self.n_inputs = n_inputs
        self.inputs_nfeat = inputs_nfeat
        self.inputs_nhid = inputs_nhid
        self.inputs_nclass = inputs_nclass
        self.dropout = dropout

        # Every single layer
        temp_in_layer = []
        temp_bet_layer = []
        for i in range(n_inputs):
            temp_in_layer.append(GCN(inputs_nfeat[i], inputs_nhid[i], inputs_nclass[i], dropout))
        self.in_layer = nn.ModuleList(temp_in_layer)

    def forward(self, xs, adjs):
        classes_labels = []
        bet_layers_output = []
        in_layers_output = []
        features = []
        # Single layer forward to find features and classes
        for i in range(self.n_inputs):
            class_labels, feature = self.in_layer[i](xs[i], adjs[i])
            classes_labels.append(class_labels)
            features.append(feature)

            temp = torch.mm(feature, torch.t(feature))
            in_layers_output.append(temp)

        return classes_labels, in_layers_output
