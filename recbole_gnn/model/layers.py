import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, GATConv
from torch_geometric.nn.aggr import Aggregation
from torch_sparse import matmul
from recbole.model.loss import BPRLoss, forwardforward_loss_fn, EmbLoss

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

# Via Namyong Park out of forwardgnn
class GNNConv(torch.nn.Module):
    def __init__(self, gnn_type: str, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if "SAGE".lower() == gnn_type.lower():
                self.gnn = SAGEConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")
        elif "GCN".lower() == gnn_type.lower():
                self.gnn = GCNConv(in_channels=in_channels, out_channels=out_channels)
        elif "BiGNNConv".lower() == gnn_type.lower():
                self.gnn = BiGNNConv(in_channels=in_channels, out_channels=out_channels)
        elif "LightGCN".lower() == gnn_type.lower():
                self.gnn = LightGCNConv(dim=in_channels)
        elif "GAT".lower() == gnn_type.lower():
            heads = 4
            assert out_channels % heads == 0, (out_channels, heads)
            self.gnn = GATConv(in_channels=in_channels, out_channels=out_channels // heads, heads=heads)
        else:
            raise ValueError(f"Unavailable gnn: {gnn_type}")

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        #return self.gnn(x, edge_index, edge_weight)
        return self.relu(self.gnn(x, edge_index, edge_weight))

class BaseForwardLayer(nn.Module):
    def forward_train(self, x, theta,**kwargs,):
        raise NotImplementedError

    def forward_predict(self, x, edge_index, edge_label_index, theta):
        raise NotImplementedError

    @property
    def requires_training(self):
        return True

class GNNForwardLayer(BaseForwardLayer):
    def __init__(self, gnn_layer: torch.nn.Module, aggr: Aggregation, forward_learning_type: str, n_user: int, n_items: int):
        super(GNNForwardLayer, self).__init__()
        self.gnn_layer = gnn_layer
        self.layer_aggregation = aggr
        self.forward_learning_type = forward_learning_type
        self.n_users = n_user
        self.n_items = n_items
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bpr_loss = BPRLoss()
        self.relu = torch.nn.ReLU()
        self.reg_loss = EmbLoss()

    def forward(self, x, edge_index, edge_weight = None):
        return self.gnn_layer(x, edge_index, edge_weight)

    def get_embeddings(self, embeddings, edge_index, edge_weight = None):
        embeddings_list = [embeddings]

        all_embeddings =  self.forward(embeddings, edge_index, edge_weight)

        # FIXME: torch.stack only for same dimensional layers such as in LightGCN
        embeddings_list.append(all_embeddings)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = self.layer_aggregation(all_embeddings, dim=1)
        all_embeddings = torch.squeeze(all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    @staticmethod
    def link_predict(u_embeddings, i_embeddings):
        return torch.mul(u_embeddings, i_embeddings)

    def forwardlearn_bpr_loss(self, pos_scores, neg_scores):
        loss = self.bpr_loss(pos_scores, neg_scores)  # shape=(# pos and neg edges,)
        loss_mean = loss.mean()

        with torch.no_grad():
            cumulated_logits_pos = pos_scores.exp().mean().item()
            cumulated_logits_neg = (1 - neg_scores.exp()).mean().item()

        return loss_mean, (cumulated_logits_pos, cumulated_logits_neg)

    def forwardlearn_bce_loss(self, pos_scores, neg_scores):
        stacked_out = torch.stack([pos_scores, neg_scores], dim=0)
        stacked_target = torch.stack([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        loss = self.bce_with_logits_loss(stacked_out, stacked_target)  # shape=(# pos and neg edges,)
        loss_mean = loss.mean()

        pos_loss, neg_loss = torch.split(loss, loss.size(0) // 2, dim=0)

        with torch.no_grad():
            cumulated_logits_pos = pos_loss.exp().mean().item()
            cumulated_logits_neg = (1 - neg_loss.exp()).mean().item()

        return loss_mean, (cumulated_logits_pos, cumulated_logits_neg)

    @classmethod
    def ff_loss(cls, pos_scores, neg_scores, theta):
        loss_pos, cumulated_logits_pos = forwardforward_loss_fn(pos_scores, theta, target=1.0)
        loss_neg, cumulated_logits_neg = forwardforward_loss_fn(neg_scores, theta, target=0.0)
        loss = loss_pos + loss_neg

        return loss, (cumulated_logits_pos, cumulated_logits_neg)

    def forward_train(self, embeddings, user, pos_item, neg_item, edge_index, edge_weight, theta, **kwargs):

        user_all_embeddings, item_all_embeddings = self.get_embeddings(embeddings, edge_index, edge_weight)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        out_pos = self.link_predict(u_embeddings, pos_embeddings)  # shape=(# pos and neg edges, node-emb-dim)
        out_neg = self.link_predict(u_embeddings, neg_embeddings)

        if self.forward_learning_type == "FL-BPR":
            loss, logit_tuple = self.forwardlearn_bpr_loss(out_pos, out_neg)
        elif self.forward_learning_type == "FL-BCE":
            loss, logit_tuple = self.forwardlearn_bce_loss(out_pos, out_neg)
        elif self.forward_learning_type == "FF":
            loss, logit_tuple = self.ff_loss(out_pos, out_neg, theta)
        else:
            raise ValueError(f"Undefined: {self.forward_learning_type}")

        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        loss = loss #+ reg_loss

        return loss, logit_tuple

    @torch.no_grad()
    def forward_predict(self, x, edge_index, edge_label_index, theta,):
        """Evaluate the layer with the given input and theta."""
        node_emb = self.forward(x, edge_index)
        out = self.link_predict(node_emb, edge_label_index).sum(dim=1)

        edge_score = out.sigmoid()

        return node_emb, edge_score


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
