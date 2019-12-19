from __future__ import division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
__author__ = 'Marc, Dini'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges, with_bias=True, use_bn=False):
        super(GraphConv, self).__init__()
        self.C = output_dim
        self.L = num_edges
        self.F = input_dim
        self.gpu = torch.cuda.is_available()
        # h_weights: (L+1) (type of edges), F(input features), c(output dim)
        self.h_weights = nn.Parameter(
            torch.FloatTensor(self.F*(self.L+1), self.C))
        self.bias = nn.Parameter(
            torch.FloatTensor(self.C)) if with_bias else None
        # Todo: init the weight
        nn.init.xavier_normal_(self.h_weights)
        nn.init.normal_(self.bias, mean=0.0001, std=0.00005)
        self.use_bn = use_bn


    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

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
        if self.use_bn:
            return self.apply_bn(F.relu(V_out).view(B, N, self.C))
        else:
            return F.relu(V_out).view(B, N, self.C)

class LinearEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim,
                 with_bias=True, with_bn=False,
                 with_act_func=True):
        super(LinearEmbedding, self).__init__()
        self.gpu = torch.cuda.is_available()
        self.embedding_axis = -1
        self.C = output_dim
        self.F = input_dim
        self.use_bn = with_bn

        self.weights = nn.Parameter(
            torch.FloatTensor(self.F, self.C))
        self.bias = nn.Parameter(
            torch.FloatTensor(self.C)) if with_bias else None
        self.relu = torch.nn.ReLU() if with_act_func else None
        nn.init.xavier_normal_(self.weights)
        nn.init.normal_(self.bias, mean=0.0001, std=0.00005)


    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def forward(self, V):
        V = torch.matmul(V, self.weights) + self.bias.unsqueeze(0).unsqueeze(0)
        if self.use_bn:
            V = self.apply_bn(V)
        if self.relu is not None:
            return self.relu(V)
        else:
            return V



class NodeSelfAtten(nn.Module):
    def __init__(self, input_dim):
        super(NodeSelfAtten, self).__init__()
        self.F = input_dim
        self.f = LinearEmbedding(input_dim, int(self.F//8))
        self.g = LinearEmbedding(input_dim, int(self.F//8))
        self.h = LinearEmbedding(input_dim, self.F)
        # Default tf softmax is -1, default torch softmax is flatten
        self.softmax = torch.nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.FloatTensor(input_dim))

    def forward(self, V):
        B = list(V.size())[0]
        # print("Inp selfatten V: ", V.size())
        f_out = self.f(V) # B x N X F//8
        g_out = self.g(V).transpose(1, 2) # B x F//8 x N
        h_out = self.h(V) # B x N x F
        s = self.softmax(torch.matmul(f_out, g_out)) # B x N x N
        o = torch.matmul(s, h_out)
        return  self.gamma*o + V


class RobustFilterGraphCNNConfig1(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges, net_size=256):
        super(RobustFilterGraphCNNConfig1, self).__init__()
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
        self.self_atten = NodeSelfAtten(int(self.net_size/2))

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

        new_V = self.self_atten(self.emb2(new_V))

        new_V = self.gcn5(self.dropout6(self.gcn4(self.dropout5(new_V), A)), A)
        return self.last_linear(new_V)

    def loss(self, output, target):
        return self.criterion(output.view(-1, self.output_dim), target.view(-1))
