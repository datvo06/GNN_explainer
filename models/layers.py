import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F



def make_linear_relu(input_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )




class GraphConv(nn.Module):
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
        A = A.view(B*N, N, self.L+1)  # (BN), N, (L+1)

        # Let's reverse stuffs a little bit for memory saving...
        # Since L is often much smaller than C, and we don't have that much mem
        # Aggregate node information first
        A = A.transpose(1, 2).reshape(-1, (self.L+1)*N, N) # BN(L+1), N
        new_V = torch.matmul(A, V.view(-1, N, self.F)).view(-1, N,
                                                            (self.L+1)*self.F)

        # BN, (L+1)*F
        V_out = torch.matmul(new_V, self.h_weights) + self.bias.unsqueeze(0)
        return V_out.view(B, N, self.C)

    def __repr__(self):
        return f"GraphConv(in_dim={self.F}, out_dim={self.C}, num_edges={self.L}, bias={self.bias is not None})"


class NodeSelfAtten(nn.Module):
    def __init__(self, input_dim):
        super(NodeSelfAtten, self).__init__()
        self.F = input_dim
        self.f = make_linear_relu(input_dim, int(self.F//8))
        self.g = make_linear_relu(input_dim, int(self.F//8))
        self.h = make_linear_relu(input_dim, self.F)
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


