__author__ = 'Marc, Dini'
from __future__ import division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges):
        super(GraphConv, self).__init__()
        self.C = output_dim
        self.L = num_edges
        self.F = input_dim
        # h_weights: (L+1) (type of edges), F(input features), c(output dim)
        self.h_weights = nn.Parameter(
            torch.FloatTensor(self.L+1, self.C, self.F))
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
            0)
        A = torch.cat([N, A], dim=-1) # BxNxNx(L+1)
        A = A.view(-1, N, self.L+1) # (BN), N, (L+1)
        h = self.h_weights.view(self.L+1, self.C*self.F)
        # each row of A (vetor) will be activated with h
        H = torch.matmul(A, h) # (BN) x N x (CF)
        H = H.view(-1,  N, self.C, self.F) # B, N, N, C, F
        # H storing a matrix of C pad, each contains an F activation weights
        # it's important to keep H as the prior matrix
        # inorder for the logic to be correct, if this is dirrected edge
        # Still, doing this would be fine...
        H = H.transpose(1, 2) # B, N, C, N, F)
        H = H.view(B, -1, N*self.F) # BxNCxNF
        V = V.view(-1, N*self.F)  # BxNF
        # For boardcast stuffs
        V = torch.unsqueeze(V, -1) #BxNFx1
        V_out = torch.matmul(H, V) # BxNCx1
        V_out = V.view(B, N, self.C)
        return F.relu(V_out)


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
        f_out = self.f(V, A) # B x N X F//8
        g_out = self.g(V, A).transpose(1, 2) # B x F//8 x N
        h_out = self.h(V, A) # B x N x F
        s = self.softmax(torch.matmul(f_out, g_out)) # B x N x N
        o = torch.matmul(s, h_out)
        return  self.gamma*o + V


class RobustFilterGraphCNNConfig1(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges):
        super(RobustFilterGraphCNNConfig1, self).__init__()
        self.gcn1 = GraphConv(input_dim, 128, num_edges)
        self.dropout1 = torch.nn.modules.Dropout(p=0.5)
        self.gcn2 = GraphConv(128, 128, num_edges)
        self.dropout2 = torch.nn.modules.Dropout(p=0.5)

        self.gcn3 = GraphConv(128, 128, num_edges)
        self.dropout3 = torch.nn.modules.Dropout(p=0.5)

        self.gcn4 = GraphConv(256, 128, num_edges)
        self.dropout4 = torch.nn.modules.Dropout(p=0.5)

        self.gcn5 = GraphConv(128, 64, num_edges)
        self.self_atten = NodeSelfAtten(128, num_edges)

        self.dropout5 = torch.nn.modules.Dropout(p=0.5)
        self.gcn6 = GraphConv(128, 64, num_edges)
        self.dropout6 = torch.nn.modules.Dropout(p=0.5)
        self.gcn7 = GraphConv(64, 32, num_edges)
        self.last_linear = torch.nn.Linear(
            in_features=32, out_features=output_dim, bias=True)

    def forward(self, V, A):
        g1 = self.dropout2(self.gcn2(self.dropout1(self.gcn1(V,A)), A))
        g2 = self.dropout3(self.gcn3(g1, A))
        new_V = torch.cat([g2, g1], dim=-1)
        g3 = self.dropout4(self.gcn4(new_V, A))
        new_V = torch.cat([g3, g1], dim=-1)
        new_V = self.self_atten(self.gcn5(new_V, A), A)

        new_V = self.gcn7(self.dropout6(self.gcn6(self.dropout5(new_V), A)), A)
        return self.last_linear(new_V.view(-1, 32))
