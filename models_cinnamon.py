from __future__ import division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
__author__ = 'Marc, Dini'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges, with_bias=True):
        super(GraphConv, self).__init__()
        self.C = output_dim
        self.L = num_edges
        self.F = input_dim
        # h_weights: (L+1) (type of edges), F(input features), c(output dim)
        self.h_weights = nn.Parameter(
            torch.FloatTensor(self.F*(self.L+1), self.C))
        self.bias = nn.Parameter(
            torch.FloatTensor(self.C)) if with_bias else None
        # Todo: init the weight

    def forward(self, V, A):
        """
        Args:
            V: BxNxF
            A: BxNxNxL
        """
        B = list(A.size())[0]
        N = list(A.size())[1]
        I = torch.unsqueeze(
            torch.unsqueeze(torch.eye(N), -1),
            0).to(device)
        A = torch.cat([I, A], dim=-1)  # BxNxNx(L+1)
        A = A.view(B*N, N, self.L+1)  # (BN), N, (L+1)

        # Let's reverse stuffs a little bit for memory saving...
        # Since L is often much smaller than C, and we don't have that much mem
        # Aggregate node information first
        A = A.transpose(1, 2).reshape(-1, (self.L+1)*N, N) # BN(L+1), N
        new_V = torch.matmul(A, V.view(-1, N, self.F)).view(-1, N,
                                                            (self.L+1)*self.F)
        # BN, (L+1)*F
        V_out = torch.matmul(new_V, self.h_weights) + self.bias.unsqueeze(0)
        return F.relu(V_out).view(B, N, self.C)


class NodeSelfAtten(nn.Module):
    def __init__(self, input_dim, num_edges):
        super(NodeSelfAtten, self).__init__()
        self.F = input_dim
        self.f = GraphConv(input_dim, self.F//8, num_edges)
        self.g = GraphConv(input_dim, self.F//8, num_edges)
        self.h = GraphConv(input_dim, self.F, num_edges)
        # Default tf softmax is -1, default torch softmax is flatten
        self.softmax = torch.nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.FloatTensor(input_dim))

    def forward(self, V, A):
        B = list(V.size())[0]
        # print("Inp selfatten V: ", V.size())
        f_out = self.f(V, A) # B x N X F//8
        g_out = self.g(V, A).transpose(1, 2) # B x F//8 x N
        h_out = self.h(V, A) # B x N x F
        s = self.softmax(torch.matmul(f_out, g_out)) # B x N x N
        o = torch.matmul(s, h_out)
        return  self.gamma*o + V


class RobustFilterGraphCNNConfig1(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges):
        super(RobustFilterGraphCNNConfig1, self).__init__()
        self.output_dim = output_dim
        self.gcn1 = GraphConv(input_dim, 128, num_edges)
        self.dropout1 = torch.nn.modules.Dropout(p=0.5)
        self.gcn2 = GraphConv(128, 128, num_edges)
        self.dropout2 = torch.nn.modules.Dropout(p=0.5)

        self.gcn3 = GraphConv(128, 128, num_edges)
        self.dropout3 = torch.nn.modules.Dropout(p=0.5)

        self.gcn4 = GraphConv(256, 128, num_edges)
        self.dropout4 = torch.nn.modules.Dropout(p=0.5)

        self.gcn5 = GraphConv(256, 64, num_edges)
        self.self_atten = NodeSelfAtten(64, num_edges)

        self.dropout5 = torch.nn.modules.Dropout(p=0.5)
        self.gcn6 = GraphConv(64, 64, num_edges)
        self.dropout6 = torch.nn.modules.Dropout(p=0.5)
        self.gcn7 = GraphConv(64, 32, num_edges)
        self.last_linear = torch.nn.Linear(
            in_features=32, out_features=output_dim, bias=True)

    def forward(self, V, A):
        g1 = self.dropout2(self.gcn2(self.dropout1(self.gcn1(V,A)), A))
        g2 = self.dropout3(self.gcn3(g1, A))
        new_V = torch.cat([g2, g1], dim=-1)
        # print(new_V.size())

        g3 = self.dropout4(self.gcn4(new_V, A))
        # print("Here\n")

        new_V = torch.cat([g3, g1], dim=-1)
        # print("new V: ", new_V.size())

        new_V = self.self_atten(self.gcn5(new_V, A), A)

        new_V = self.gcn7(self.dropout6(self.gcn6(self.dropout5(new_V), A)), A)
        return self.last_linear(new_V.view(-1, 32))


    def loss(self, output, target):
        pred = F.log_softmax(output.view(-1, self.output_dim), dim=-1)
        loss = F.nll_loss(pred, target.view(-1))
        return loss
