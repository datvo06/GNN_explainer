from __future__ import division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import init
from models_cinnamon import *
import torch.nn.functional as F

import numpy as np
__author__ = 'Marc, Dini'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class FUNSDModelConfig1(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges):
        super(FUNSDModelConfig1, self).__init__()
        self.output_dim = output_dim
        self.net_size = 512
        self.emb1 = LinearEmbedding(input_dim, self.net_size)
        self.dropout1 = torch.nn.modules.Dropout(p=0.5)
        self.emb2 = LinearEmbedding(self.net_size, self.net_size)
        self.dropout2 = torch.nn.modules.Dropout(p=0.5)
        self.emb3 = LinearEmbedding(self.net_size, self.output_dim)
        self.dropout3 = torch.nn.modules.Dropout(p=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, V, A):
        new_V = self.dropout1(self.emb1(V))
        new_V = self.dropout2(self.emb2(new_V))
        new_V = self.dropout3(self.emb3(new_V))
        return new_V

    def loss(self, output, target):
        return self.criterion(output.view(-1, self.output_dim), target.view(-1))


class FUNSDModelConfig2(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges, net_size=256):
        super(FUNSDModelConfig2, self).__init__()
        self.output_dim = output_dim
        self.net_size = net_size
        self.emb1 = LinearEmbedding(input_dim, self.net_size)
        self.dropout1 = torch.nn.modules.Dropout(p=0.5)

        self.gcn1 = GraphConv(self.net_size, self.net_size, num_edges)
        self.dropout2 = torch.nn.modules.Dropout(p=0.5)

        self.gcn2 = GraphConv(self.net_size, self.net_size, num_edges)
        self.dropout3 = torch.nn.modules.Dropout(p=0.5)

        self.gcn3 = GraphConv(self.net_size*2, self.net_size, num_edges)
        self.dropout4 = torch.nn.modules.Dropout(p=0.5)

        self.emb2 = LinearEmbedding(self.net_size*2, int(self.net_size/2))

        self.dropout5 = torch.nn.modules.Dropout(p=0.5)
        self.gcn4 = GraphConv(int(self.net_size/2), int(self.net_size/2), num_edges)
        self.dropout6 = torch.nn.modules.Dropout(p=0.5)
        self.gcn5 = GraphConv(int(self.net_size/2), int(self.net_size/2), num_edges)
        self.last_linear = LinearEmbedding(
            int(self.net_size/2), output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, V, A):
        g1 = self.dropout2(self.gcn1(self.dropout1(self.emb1(V)), A))
        g2 = self.dropout3(self.gcn2(g1, A))
        new_V = torch.cat([g2, g1], dim=-1)
        # print(new_V.size())

        g3 = self.dropout4(self.gcn3(new_V, A))
        # print("Here\n")

        new_V = torch.cat([g3, g1], dim=-1)
        # print("new V: ", new_V.size())

        new_V = self.emb2(new_V)

        new_V = self.gcn5(self.dropout6(self.gcn4(self.dropout5(new_V), A)), A)
        return self.last_linear(new_V)

    def loss(self, output, target):
        return self.criterion(output.view(-1, self.output_dim), target.view(-1))
