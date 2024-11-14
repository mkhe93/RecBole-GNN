# @Time   : 2022/3/8
# @Author : Lanling Xu
# @Email  : xulanling_sherry@163.com

r"""
LightGCN
################################################
Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import torch

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from torch_geometric.nn import Sequential, Linear
from torch_geometric.nn.aggr import MeanAggregation, SoftmaxAggregation, LSTMAggregation, AttentionalAggregation, GRUAggregation

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv, GNNForwardLayer, GNNConv


class ForwardGNN(GeneralGraphRecommender):
    r"""LightGCN is a GCN-based recommender model, implemented via PyG.
    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ForwardGNN, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.out_channels = config['out_channels']
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization
        self.gnn_type = config['gnn_type']  # str type: which kind of gnn to use as layer
        self.pre_model_path = config['pre_model_path'] # needed for finetune

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        # FIXME: allow differing channel sizes dynamically!
        """        self.forward_convs = torch.nn.ModuleList(
            [GNNForwardLayer(
                GNNConv(self.gnn_type, in_channels=self.latent_dim, out_channels=self.out_channels),
                config['forward_learning_type'],
                self.n_users,
                self.n_items)
                for _ in range(self.n_layers)])"""


        self.forward_convs = torch.nn.ModuleList(
        [GNNForwardLayer(
            GNNConv(self.gnn_type, in_channels=128, out_channels=128),
            MeanAggregation(),
            config['forward_learning_type'],
            self.n_users,
            self.n_items),
        GNNForwardLayer(
            GNNConv(self.gnn_type, in_channels=128, out_channels=128),
            MeanAggregation(),
            config['forward_learning_type'],
            self.n_users,
            self.n_items)
        ])

        gate_nn = torch.nn.Sequential(
            Linear(128, 64),
            torch.nn.ReLU(),
            Linear(64, 1)
        )

        # Define nn to transform embeddings to the output dimension
        feature_nn = torch.nn.Sequential(
            Linear(128, 128),
            torch.nn.ReLU()
        )

        #self.aggregation = MeanAggregation()
        #self.aggregation = SoftmaxAggregation(learn=True, channels=1)
        self.aggregation = AttentionalAggregation(gate_nn=gate_nn, nn=feature_nn)
        #self.aggregation = GRUAggregation(in_channels=128, out_channels=128)
        #self.aggregation = LSTMAggregation(in_channels=128, out_channels=128)


        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.train_stage = config["train_stage"]

        # parameters initialization
        assert self.train_stage in ["pretrain", "finetune"]
        if self.train_stage == "pretrain":
            self.apply(xavier_uniform_initialization)
            # self.aggregation.reset_parameters()
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            self.load_state_dict(pretrained["state_dict"])

        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_splitted_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer in self.forward_convs:
            all_embeddings = layer(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)

        if isinstance(self.aggregation, GRUAggregation) or isinstance(self.aggregation, LSTMAggregation):
            aggregated_embeddings = []
            # Final Aggregation of layer outputs
            for node_embeddings in lightgcn_all_embeddings:  # Each `node_embeddings` is [3, 128]
                node_embedding_aggregated = self.aggregation(node_embeddings)  # Output shape: [64]
                aggregated_embeddings.append(node_embedding_aggregated)

            lightgcn_all_embeddings = torch.cat(aggregated_embeddings, dim=0)
        else:
            lightgcn_all_embeddings = self.aggregation(lightgcn_all_embeddings)
            lightgcn_all_embeddings = torch.squeeze(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def get_layer_train_data(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        embeddings = self.get_ego_embeddings()

        return embeddings, user, pos_item, neg_item

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def predict_per_layer(self, interaction, layer_num):
        """Evaluate the layer with the given input and theta."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward_convs[layer_num].get_embeddings(self.get_ego_embeddings(), self.edge_index, self.edge_weight)
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = self.link_predict(u_embeddings, i_embeddings).sum(dim=1)
        edge_score = scores

        return edge_score

    def full_sort_predict_per_layer(self, interaction, layer_num):
        """Evaluate the layer with the given input and theta."""
        user = interaction[self.USER_ID]

        #if self.restore_user_e is None or self.restore_item_e is None:
        self.restore_user_e, self.restore_item_e = self.forward_convs[layer_num].get_embeddings(self.get_ego_embeddings(), self.edge_index, self.edge_weight)

        # TOOD: print the weights of each layer here!
        #print(self.forward_convs[layer_num].gnn_layer.gnn.weight)

        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1)).view(-1)

        return scores