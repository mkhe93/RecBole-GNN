import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv
from torch_sparse import matmul
from recbole.model.loss import BPRLoss

# Via Namyong Park out of forwardgnn
class GNNConv(torch.nn.Module):
    def __init__(self, gnn_type, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if "SAGE".lower() in gnn_type.lower():
                self.gnn = SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")
        elif "GCN".lower() in gnn_type.lower():
                self.gnn = GCNConv(in_channels=in_channels, out_channels=out_channels)
        elif "LightGCN".lower() in gnn_type.lower():
                self.gnn = LightGCNConv(dim=in_channels)
        elif "GAT".lower() in gnn_type.lower():
            heads = 4
            assert out_channels % heads == 0, (out_channels, heads)
            self.gnn = GATConv(in_channels=in_channels, out_channels=out_channels // heads, heads=heads)
        else:
            raise ValueError(f"Unavailable gnn: {gnn_type}")

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        return self.gnn(x, edge_index, edge_weight)

        #return self.relu(self.gnn(x, edge_index))




class BaseForwardLayer(nn.Module):
    def forward_train(self, x, theta,**kwargs,):
        raise NotImplementedError

    def forward_predict(self, x, edge_index, edge_label_index, theta):
        raise NotImplementedError

    @property
    def requires_training(self):
        return True

class GNNForwardLayer(BaseForwardLayer):
    def __init__(self, gnn_layer):
        super(GNNForwardLayer, self).__init__()
        #self.gnn_layer = SAGEConv(in_channels=hidden_channels, out_channels=out_channels, aggr="mean")
        #self.gnn_layer = GATConv(in_channels=hidden_channels, out_channels=out_channels // 4, heads=4)
        #self.gnn_layer = GCNConv(in_channels=hidden_channels, out_channels=out_channels)
        #self.gnn_layer = LightGCNConv(dim=hidden_channels)
        self.gnn_layer = gnn_layer
        self.criterion_no_reduction = BPRLoss()

    def forward(self, x, edge_index, edge_weight = None):
        return self.gnn_layer(x, edge_index, edge_weight)

    @staticmethod
    def link_predict(u_embeddings, i_embeddings):
        return torch.mul(u_embeddings, i_embeddings).sum(dim=1)

    def forwardlearn_loss(self, pos_scores, neg_scores):
        loss = self.criterion_no_reduction(pos_scores, neg_scores)  # shape=(# pos and neg edges,)
        loss_mean = loss.mean()

        with torch.no_grad():
            cumulated_logits_pos = pos_scores.exp().mean().item()
            cumulated_logits_neg = (1 - neg_scores.exp()).mean().item()

        return loss_mean, (cumulated_logits_pos, cumulated_logits_neg)

    def forward_train(self, embeddings, u_embeddings, pos_embeddings, neg_embeddings, edge_index, edge_weight, theta, **kwargs):
        x = embeddings
        x = self.forward(x, edge_index, edge_weight)
        out_pos = self.link_predict(u_embeddings, pos_embeddings)  # shape=(# pos and neg edges, node-emb-dim)
        out_neg = self.link_predict(u_embeddings, neg_embeddings)
        loss, logit_tuple = self.forwardlearn_loss(out_pos, out_neg)

        return loss, logit_tuple

    @torch.no_grad()
    def forward_predict(self, x, edge_index, edge_label_index, theta,):
        """Evaluate the layer with the given input and theta."""
        node_emb = self.forward(x, edge_index)
        out = self.link_predict(node_emb, edge_label_index)

        edge_score = out.sum(dim=1).sigmoid()

        return node_emb, edge_score



class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)

class BipartiteGCNConv(MessagePassing):
    def __init__(self, dim):
        super(BipartiteGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
