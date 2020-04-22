import torch
from torch import nn

from .layers import add_self_loop_, get_degree_inverse_from_
from .layers import GraphConvolution

class GraphNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_classes,
                 activation='relu',
                 p_dropout=0.3,
                 n_adj_matrix=1
                ):
        super(GraphNN, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        if isinstance(activation, str):
            if activation == 'relu':
                activation = nn.ReLU()
            else:
                raise ValueError("Not supported activation type")
        
        self.activation = activation
        self.gcn1 = GraphConvolution(n_in, n_hidden)
        self.gcn2 = GraphConvolution(n_hidden * n_adj_matrix, n_classes)
        self.dropout = nn.Dropout(p_dropout)


    def forward(self, x, a, process_a=False):
        if process_a:
            a = add_self_loop_(a)
            d_inverse = get_degree_inverse_from_(a)
            d_times_a = d_inverse * a
        else:
            d_times_a = a


        z = self.activation(self.gcn1(x, d_times_a))
        z = self.dropout(z)
        return self.gcn2(z, d_times_a)


# UNIT TEST
if __name__ == "__main__":
    n_batch = 3
    n_in = 5
    n_hidden = 7
    n_classes = 10 
    n_vertex = 6

    x = torch.rand(n_batch, n_vertex, n_in)
    a = torch.rand(n_batch, n_vertex, n_vertex)

    model = GraphNN(n_in, n_hidden, n_classes)

    assert model(x, a).shape == (n_batch, n_vertex, n_classes)



    