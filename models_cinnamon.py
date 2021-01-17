from __future__ import division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
__author__ = 'Marc, Dini, Xing'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_linear_relu(input_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )


class RobustFilterGraphCNNConfig1(nn.Module):
    """ The Core of Graph KV Module created by Ethan in tensorflow

    Using the original name as tribute to him and Marc, Dini
    who is the original writer of this model

    """
    def __init__(self, input_dim, output_dim, num_edges, net_size=256, use_self_atten=False):
        super(RobustFilterGraphCNNConfig1, self).__init__()
        self.output_dim = output_dim
        self.net_size = net_size
        self.emb1 = make_linear_relu(input_dim, self.net_size)

        self.dropout = nn.Dropout(p=0.5)
        self.gcn1 = GraphConv(self.net_size, self.net_size, num_edges)
        self.gcn2 = GraphConv(self.net_size, self.net_size, num_edges)

        self.gcn3 = GraphConv(self.net_size * 2, self.net_size, num_edges)

        half_net_size = self.net_size // 2
        self.emb2 = make_linear_relu(self.net_size * 2, half_net_size)

        self.self_atten = NodeSelfAtten(half_net_size)

        self.gcn4 = GraphConv(half_net_size, half_net_size, num_edges)
        self.gcn5 = GraphConv(half_net_size, half_net_size, num_edges)

        self.classifier = nn.Linear(half_net_size, output_dim)
        self.use_self_atten = use_self_atten
        self.criterion = torch.nn.CrossEntropyLoss()


    def forward(self, V, A):
        embedding = self.dropout(self.emb1(V))

        # First GraphConv
        g1 = F.relu(self.gcn1(embedding, A))
        g1 = self.dropout(g1)

        # Second GraphConv
        g2 = F.relu(self.gcn2(g1, A))
        g2 = self.dropout(g2)

        # Third GraphConv
        new_v = torch.cat([g2, g1], dim=-1)
        g3 = F.relu(self.gcn3(new_v, A))
        g3 = self.dropout(g3)

        new_v = torch.cat([g3, g1], dim=-1)
        if self.use_self_atten:
          new_v = self.self_atten(self.emb2(new_v))
        else:
          new_v = self.emb2(new_v)

        new_v = self.dropout(new_v)

        # Final feature extractor
        g4 = F.relu(self.gcn4(new_v, A))
        g4 = self.dropout(g4)
        g5 = F.relu(self.gcn5(g4, A))

        return self.classifier(g5)

    def loss(self, output, target):
        return self.criterion(output.view(-1, self.output_dim),
                              target.view(-1))

class GraphConv(nn.Module):
    '''Written by Marc, Dini, standardized by Xing'''
    def __init__(self, input_dim, output_dim, num_edges, with_bias=True):
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

    def forward(self, V, A):
        """
        Args:
            V: BxNxF
            A: BxNxNxL
        """
        B = list(A.size())[0]
        N = list(A.size())[1]
        identity_matrix = torch.unsqueeze(torch.eye(N), -1)
        I = torch.stack([identity_matrix for i in range(B)])

        # Dirty way to get device
        cur_device = next(self.parameters()).device
        I = I.to(cur_device)


        A = torch.cat([I, A], dim=-1)  # BxNxNx(L+1)
        A = A.transpose(1, 3).transpose(2, 3)
        A = A.view(B*(self.L+1), N, N)  # (B(L+1)), N, N

        # Let's reverse stuffs a little bit for memory saving...
        # Since L is often much smaller than C, and we don't have that much mem
        # Aggregate node information first
        new_V = torch.matmul(
            A, V.view(-1, N, self.F)).view(
                B, self.L+1, N, self.F
            ).transpose(1, 2).contiguous().view(B, N, -1)  # B(L+1) N F => B N (L+1)F

        # BN, (L+1)*F
        V_out = torch.matmul(new_V, self.h_weights) + self.bias.unsqueeze(0)
        return V_out.view(B, N, self.C)

    def __repr__(self):
        return f"GraphConv(in_dim={self.F}, out_dim={self.C}, num_edges={self.L}, bias={self.bias is not None})"



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
        '''
        V: 1, N, F
        '''
        V = torch.matmul(V, self.weights) + self.bias.unsqueeze(0).unsqueeze(0)
        if self.use_bn:
            V = self.apply_bn(V)
        if self.relu is not None:
            return self.relu(V)
        else:
            return V


class LEConvMultiEdge(nn.Module):
    def __init__(self, input_dim, output_dim, num_edges, with_bias=True, use_bn=False):
        super(LEConvMultiEdge, self).__init__()
        self.C = output_dim
        self.L = num_edges
        self.F = input_dim
        self.gpu = torch.cuda.is_available()
        self.w1 = LinearEmbedding(self.F, self.C, with_bias=False,
                                  with_bn=False, with_act_func=False)
        self.w2 = nn.Parameter(
            torch.FloatTensor(self.F*self.L, self.C))
        self.w3 = nn.Parameter(
            torch.FloatTensor(self.F*self.L, self.C))

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
            V: B x N x F
            A: B x N x N x L
        """
        B = list(A.size())[0]
        N = list(A.size())[1]

        term1 = self.w1(V) # B x N x C

        sum_A = torch.squeeze(torch.sum(A, dim=2))    # B x N x L
        sum_A = torch.transpose(sum_A, (1, 2))      # B x L x N
        sum_A = torch.unsqueeze(sum_A, -1)      # B x L x N x 1
        # Broadcast V
        V2 = torch.unsqueeze(V, -1)         # B x 1 x N x F
        term2 = torch.transpose(sum_A * V2, (1, 2)).view(-1, self.L * self.F) # B x L x N x F => (B x N) x (L x F)
        term2 = torch.matmul(term2, self.w2.view(B, N, self.C))  # B x N x C

        new_A = torch.transpose(torch.transpose(A, (1, 3)), (2, 3))  # B x L x N x N, transpose twice to keep same meaning
        new_A = torch.matmul(new_A, V)              # B x L x N x F
        new_A = torch.transpose(new_A, (1, 2)).view(B, N, self.L*self.F) # B x N x (L x F)
        term3 = torch.matmul(new_A, self.w3) # B x N x C

        return torch.sigmoid(term1 + term2 - term3)


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
        f_out = self.f(V)  # B x N X F//8
        g_out = self.g(V).transpose(1, 2)  # B x F//8 x N
        h_out = self.h(V)  # B x N x F
        s = self.softmax(torch.matmul(f_out, g_out))  # B x N x N
        o = torch.matmul(s, h_out)
        return self.gamma*o + V


class MLPNodeLink(nn.Module):
    def __init__(self, input_dim, hiddens=[512, 512],
                 activation=torch.nn.ReLU):
        super(MLPNodeLink, self).__init__()
        self.F = input_dim*2
        self.hiddens = torch.nn.ModuleList()
        for i, hidden_dim in enumerate(hiddens):
            if i == 0:
                hidden1 = self.F
            else:
                hidden1 = hiddens[i-1]
            hidden2 = hidden_dim
            self.hiddens.append(LinearEmbedding(hidden1, hidden2))
        self.hiddens.append(LinearEmbedding(hiddens[-1], 2))

    def forward(self, V1, V2):
        """ The prediction is anisotropic/non-symmetric
        V1: 1xN1xF
        V2: 1xN2xF
        """
        N1 = list(V1.size())[1]
        N2 = list(V2.size())[1]
        new_V = []
        for i in range(N1):
            for j in range(N2):
                # V1[0][i]: F
                new_V.append(torch.cat((V1[0][i], V2[0][j]), dim=-1))
        # new V: (N1*N2) x2F
        new_V = torch.stack(new_V).unsqueeze(0)  # (1xN1xN2, 2F)
        for hidden in self.hiddens:
            new_V = hidden(new_V)
        # new V: (1, N1xN2, 2)
        # Each of V1 will have N2 prediction
        return new_V
