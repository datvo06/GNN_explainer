from .layers import GraphConv, NodeSelfAtten, make_linear_relu
from torch import nn
from torch.nn import functional as F
import torch

class RobustFilterGraphCNNConfig1(nn.Module):
    """ The Core of Graph KV Module created by Ethan
    
    Using the original name as tribute to him and Marc, Dini
    who is the original writer of this source code
    
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
        



