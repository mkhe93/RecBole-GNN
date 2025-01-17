# @Time   : 2024/1/17
# @Author : Markus Hoefling
# @Email  : markus.hoefling01@gmail.com

r"""
UltraGCN
################################################
Reference:
    Kelong, Mao et al. "UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation" in CIKM 2021.

Reference code:
    https://reczoo.github.io/UltraGCN
"""

import numpy as np
import torch
import time

from recbole.utils import InputType
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender

class UltraGCN(GeneralGraphRecommender):
    r"""UltraGCN resorts to directly approximate the limit of infinite-layer graph convolutions via a constraint loss.
     Meanwhile, UltraGCN allows for more appropriate edge weight assignments and flexible adjustment of the relative
     importances among different types of relationships. This finally yields a simple yet effective UltraGCN model,
     which is easy to implement and efficient to train. Experimental results on four benchmark datasets show
    that UltraGCN not only outperforms the state-of-the-art GCN models but also achieves more than 10x speedup over
    LightGCN. (c.f. Abstract of Reference above)
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(UltraGCN, self).__init__(config, dataset)

        self.user_num = self.n_users
        self.item_num = self.n_items
        self.embedding_dim = config['embedding_size']
        self.ii_neighbor_num = config['ii_neighbor_num']

        self.w1 = config['w1']
        self.w2 = config['w2']
        self.w3 = config['w3']
        self.w4 = config['w4']

        self.negative_num = config['negative_num']
        self.negative_weight = config['negative_weight']
        self.gamma = config['gamma']
        self.lambda_ = config['lambda']

        # define layers and loss
        self.user_embeds = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_dim)
        self.item_embeds = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_dim)

        self.constraint_mat = self.get_constraint_mat(dataset.inter_matrix(form="csr"))

        self.ii_neighbor_mat, self.ii_constraint_mat = self.get_ii_constraint_mat(dataset.inter_matrix(form="csr"),
                                                                                  self.ii_neighbor_num)

        self.initial_weight = config['initial_weight']
        torch.nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        torch.nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)
        self.user_embeds.to(self.device)
        self.item_embeds.to(self.device)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embeds.weight
        item_embeddings = self.item_embeds.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        device = self.get_device()

        # get all users, pos_items out of the batch
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]

        # Negative sampling strategy by the authors
        # TODO: exclude the negative sampling into a separate RecBole sampler
        neg_candidates = np.arange(self.item_num)
        neg_items = np.random.choice(neg_candidates, (len(users), self.negative_num), replace=True)
        neg_items = torch.from_numpy(neg_items)

        # get embeddings
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))

        # calculate helpers for loss
        sim_scores = self.ii_constraint_mat[pos_items].to(device)  # len(pos_items) * num_neighbors
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(user_embeds, pos_embeds, neg_embeds, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(neighbor_embeds, self.user_embeds(users).unsqueeze(1), sim_scores)

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        items = torch.arange(self.num_items).to(self.device)
        user_embeds = self.user_embeds(user.to(self.device))
        item_embeds = self.item_embeds(items.to(self.device))

        return user_embeds.mm(item_embeds.t())

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, item_embeddings.transpose(0, 1))

        return scores.view(-1)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(
                device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                                   self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def get_device(self):
        return self.user_embeds.weight.device

    def cal_loss_L(self, user_embeds, pos_embeds, neg_embeds, omega_weight):
        device = self.get_device()

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)            # due to different neg. sampling
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        weight = omega_weight[len(pos_scores):].view(neg_scores.size())
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores,
                                                      neg_labels,
                                                      weight=weight,
                                                      reduction='none').mean(dim=-1) # due to different neg. sampling

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)],
                                                      reduction='none')

        neg_loss = neg_loss * self.negative_weight
        loss = pos_loss + neg_loss
        loss = loss.sum()

        return loss

    def cal_loss_I(self, neighbor_embeds, user_embeds, sim_scores):
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        loss = loss.sum()
        # loss = loss.sum(-1)
        return loss

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        loss = loss / 2
        return loss

    def get_constraint_mat(self, train_mat: csr_matrix):
        # construct degree matrix
        items_D = np.array(train_mat.sum(axis=0)).reshape(-1)  # Sum over rows (axis=0)
        users_D = np.array(train_mat.sum(axis=1)).reshape(-1)  # Sum over columns (axis=1)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
        beta_uD = (np.sqrt(users_D + 1) / (users_D + epsilon)).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        # Convert to PyTorch tensors
        constraint_mat = {
            "beta_uD": torch.from_numpy(beta_uD).float().reshape(-1),
            "beta_iD": torch.from_numpy(beta_iD).float().reshape(-1)
        }

        return constraint_mat

    def get_ii_constraint_mat(self, train_mat: csr_matrix, num_neighbors: int, ii_diagonal_zero=False, block_size=100):
        """
        Compute the item-item constraint matrix using precomputed item-item similarities.

        Args:
            train_mat (csr_matrix): The training interaction matrix.
            num_neighbors (int): Number of top neighbors to retrieve for each item.
            ii_diagonal_zero (bool): Whether to zero out the diagonal in the similarity matrix.
            block_size (int): Number of items to process in each block for efficiency.

        Returns:
            torch.Tensor: Indices of top-k neighbors for each item.
            torch.Tensor: Similarity scores of top-k neighbors for each item.
        """
        print('Computing \\Omega for the item-item graph...')

        # Compute item-item similarity matrix (A = I.T * I)
        A = train_mat.T.dot(train_mat)  # A = I * I (item-item similarity matrix)
        n_items = A.shape[0]

        if ii_diagonal_zero:
            A.setdiag(0)
            A.eliminate_zeros()

        # Compute degree vectors
        items_D = np.array(A.sum(axis=0)).ravel()  # Column-wise sum (item degree)
        users_D = np.array(A.sum(axis=1)).ravel()  # Row-wise sum (user degree)

        epsilon = 1e-10
        beta_uD = (np.sqrt(users_D + 1) / (users_D + epsilon)).reshape(-1, 1)  # User constraints
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)  # Item constraints

        # Precompute the constraint matrix
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD)).float()

        # Initialize result matrices
        res_mat = torch.zeros((n_items, num_neighbors), dtype=torch.long)
        res_sim_mat = torch.zeros((n_items, num_neighbors), dtype=torch.float)

        # Block processing for memory efficiency
        start_block = 0
        while start_block < n_items:
            end_block = min(start_block + block_size, n_items)
            block_size = end_block - start_block

            # Extract the current block from the similarity matrix
            current_block = A[start_block:end_block, :].toarray()

            # Apply the constraint matrix
            constraint_block = all_ii_constraint_mat[start_block:end_block].numpy()
            current_block = constraint_block * current_block

            # Vectorized top-k computation for the entire block
            current_block_tensor = torch.from_numpy(current_block)
            row_sims, row_indices = torch.topk(current_block_tensor, num_neighbors, dim=1)

            # Save results for the current block
            res_mat[start_block:end_block] = row_indices
            res_sim_mat[start_block:end_block] = row_sims

            start_block += block_size

            if start_block % 15000 == 0:
                print(f'i-i constraint matrix {start_block} items processed')

        print('Computation \\Omega OK!')
        return res_mat, res_sim_mat

    def _custom_sampler(self, pos_train_data):
        neg_candidates = np.arange(self.item_num)
        neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), self.negative_num), replace=True)
        neg_items = torch.from_numpy(neg_items)

        return pos_train_data[0].long(), pos_train_data[1].long(), neg_items.long()  # users, pos_items, neg_items