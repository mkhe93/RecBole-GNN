import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
try:
    from torch_sparse import SparseTensor
    is_sparse = True
except ImportError:
    is_sparse = False

from recbole.utils.logger import set_color
import matplotlib.pyplot as plt
import numpy as np

import networkx
from networkx.algorithms import bipartite
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from recbole_gnn.data.dataset import GeneralGraphDataset

class GraphDatasetEvaluator(GeneralGraphDataset):
    def __init__(self, config, dataset: GeneralGraphDataset):
        super().__init__(config)

        self.inter_feat = dataset.inter_feat
        self.connected = True

        self.bipartite_graph = self.networkx_bipartite_graph()
        if not networkx.is_connected(self.bipartite_graph):
            self.connected = False
            self.make_connected()
        self.num_edges = len(self.bipartite_graph.edges)
        self.user_nodes, self.item_nodes = bipartite.sets(self.bipartite_graph)
        self.num_user_nodes, self.num_item_nodes = len(self.user_nodes), len(self.item_nodes)

        # Precompute clustering coefficients for all methods
        clustering_methods = ['dot']  # 'min', 'max'
        self.precomputed_clustering = {
            method: bipartite.clustering(self.bipartite_graph, mode=method) for method in clustering_methods
        }

    def __str__(self):
        info = [set_color(self.dataset_name, "pink")]
        info.append(set_color("Traditional graph dataset characteristics", "pink"))
        if self.uid_field:
            info.extend(
                [
                    set_color("The number of users", "blue") + f": {self.user_num}",
                    set_color("Average actions of users", "blue") + f": {self.avg_actions_of_users}",
                    set_color("Median actions of users", "blue") + f": {self.median_actions_of_users}",
                    set_color("Min actions of users", "blue") + f": {self.min_actions_of_users}",
                    set_color("Max actions of users", "blue") + f": {self.max_actions_of_users}"
                ]
            )
        if self.iid_field:
            info.extend(
                [
                    set_color("The number of items", "blue") + f": {self.item_num}",
                    set_color("Average actions of items", "blue") + f": {self.avg_actions_of_items}",
                    set_color("Median actions of items", "blue") + f": {self.median_actions_of_items}",
                    set_color("Min actions of items", "blue") + f": {self.min_actions_of_items}",
                    set_color("Max actions of items", "blue") + f": {self.max_actions_of_items}"
                ]
            )
        info.append(set_color("The number of inters", "blue") + f": {self.inter_num}")
        if self.uid_field and self.iid_field:
            info.append(
                set_color("The sparsity of the dataset", "blue")
                + f": {self.sparsity * 100}%"
            )
            info.append(set_color("Topological graph dataset characteristics", "pink"))
            info.extend(
                [
                    set_color("Sparsity log", "blue") + f": {self.sparsity_log}",
                    set_color("Space size", "blue") + f": {self.space_size}",
                    set_color("Space size log", "blue") + f": {self.space_size_log}",
                    set_color("Shape (user/item)", "blue") + f": {self.shape}",
                    set_color("Shape log", "blue") + f": {self.shape_log}",
                    set_color("Gini user", "blue") + f": {self.gini_user}",
                    set_color("Gini item", "blue") + f": {self.gini_item}",
                    set_color("Average degree", "blue") + f": {self.average_degree()}",
                    set_color("Average degree user", "blue") + f": {self.average_degree('user')}",
                    set_color("Average degree item", "blue") + f": {self.average_degree('item')}",
                    set_color("Average degree user log", "blue") + f": {self.average_degree('user', True)}",
                    set_color("Average degree item log", "blue") + f": {self.average_degree('item', True)}",
                    set_color("Degree assortativity user",
                              "blue") + f": {self.degree_assortativity(self.user_nodes)}",
                    set_color("Degree assortativity item",
                              "blue") + f": {self.degree_assortativity(self.item_nodes)}",
                    set_color("Average clustering coefficient dot",
                              "blue") + f": {self.calculate_average_clustering(method='dot', 
                                                                               nodes=self.bipartite_graph, 
                                                                               precomputed_clustering=self.precomputed_clustering)}",
                    set_color("Average clustering coefficient dot user",
                              "blue") + f": {self.calculate_average_clustering(method='dot',
                                                                               nodes=self.user_nodes,
                                                                               precomputed_clustering=self.precomputed_clustering)}",
                    set_color("Average clustering coefficient dot item",
                              "blue") + f": {self.calculate_average_clustering(method='dot',
                                                                               nodes=self.item_nodes,
                                                                               precomputed_clustering=self.precomputed_clustering)}"
                ]
            )

        info.append(set_color("Remain Fields", "blue") + f": {list(self.field2type)}")
        return "\n".join(info)

    def average_degree(self, nodes = None, log=False):
        if nodes is None:
            average_degree = (2 * self.num_edges) / (self.num_user_nodes + self.num_item_nodes)
        elif nodes == "user":
            average_degree = self.num_edges / self.num_user_nodes
        elif nodes == "item":
            average_degree = self.num_edges / self.num_item_nodes

        if log:
            average_degree = np.log10(average_degree)

        return average_degree

    def calculate_average_clustering(self, method, nodes, precomputed_clustering, log=False):
        """
        Calculate the average clustering coefficient for a subset of nodes.

        Parameters:
            method (str): The clustering method ('dot', 'min', 'max').
            nodes (list): Subset of nodes for which to calculate the average.
            precomputed_clustering (dict): Precomputed clustering coefficients.

        Returns:
            float: The average clustering coefficient for the subset of nodes.
        """
        clustering_values = precomputed_clustering[method]
        subset_clustering = [clustering_values[node] for node in nodes]
        average_clustering = sum(subset_clustering) / len(subset_clustering)

        if log:
            average_clustering = np.log10(sum(subset_clustering) / len(subset_clustering))

        return average_clustering

    def degree_assortativity(self, nodes):
        graph = bipartite.projected_graph(self.bipartite_graph, nodes)
        return networkx.degree_pearson_correlation_coefficient(graph, nodes=nodes)

    def clustering(self, mode='dot', nodes=None):
        return bipartite.clustering(self.bipartite_graph, mode=mode, nodes=nodes)

    def find_optimal_clusters(self, min_clusters=2, max_clusters=10, nodes=None):
        """
        Findet die optimale Anzahl von Clustern basierend auf der Elbow-Methode und dem Silhouetten-Score.
        Gibt die optimale Anzahl der Cluster zurück.
        """
        if nodes:
            clustering_coefficients = {
                node: self.precomputed_clustering['dot'][node]
                for node in nodes
            }
        else:
            clustering_coefficients = self.precomputed_clustering['dot']

        values = np.array(list(clustering_coefficients.values())).reshape(-1, 1)

        inertia = []
        silhouette_scores = []
        k_values = range(min_clusters, max_clusters + 1)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(values)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(values, kmeans.labels_))

        # Elbow-Methode plotten
        #plt.plot(k_values, inertia, marker='o')
        #plt.title('Elbow-Methode')
        #plt.xlabel('Anzahl der Cluster')
        #plt.ylabel('Inertia')
        #plt.show()

        # Silhouetten-Score plotten
        #plt.plot(k_values, silhouette_scores, marker='o')
        #plt.title('Silhouetten-Score')
        #plt.xlabel('Anzahl der Cluster')
        #plt.ylabel('Score')
        #plt.show()

        # Optimalen Wert auswählen (hier z. B. der maximale Silhouetten-Score)
        optimal_k = k_values[np.argmax(silhouette_scores)]
        return optimal_k

    def assign_clusters(self, n_clusters, nodes=None):
        """
        Ordnet die Knoten basierend auf den Cluster-Koeffizienten und der angegebenen Anzahl von Clustern zu.
        Gibt ein Dictionary mit den Knoten und ihren zugeordneten Clustern zurück.
        """
        if nodes:
            clustering_coefficients = {
                node: self.precomputed_clustering['dot'][node]
                for node in nodes
            }
        else:
            clustering_coefficients = self.precomputed_clustering['dot']

        values = np.array(list(clustering_coefficients.values())).reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(values)

        # Knoten den Clustern zuordnen
        cluster_assignments = dict(zip(clustering_coefficients.keys(), labels))
        return cluster_assignments

    def plot_clusters(self, cluster_assignments, nodes, graph=None, highlight_nodes=None):
        """
        Plottet nur die Benutzerknoten (user_nodes) in einem zweidimensionalen Raum.
        Optional kann ein Graph angegeben werden, um Knotengrade für die y-Achse zu verwenden.
        Optional können spezifische Knoten hervorgehoben werden.
        """
        clustering_coefficients = self.precomputed_clustering['dot']
        values = np.array([clustering_coefficients[node] for node in nodes])

        # Farben basierend auf Cluster-Zuordnung
        labels = np.array([cluster_assignments[node] for node in nodes])

        # Bestimme y-Werte basierend auf Knotengrad, falls Graph vorhanden
        if graph:
            y_values = np.array([graph.degree(node) for node in nodes])
        else:
            y_values = labels  # Falls kein Graph, Cluster als y-Wert nehmen

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(values, y_values, c=labels, cmap='viridis', s=100, alpha=0.7)
        plt.title('Cluster-Zuordnungen der Benutzerknoten')
        plt.xlabel('Cluster-Koeffizient')
        plt.ylabel('Knotengrad (y-Wert)')
        plt.colorbar(scatter, label='Cluster')

        # Bestimmte Knoten hervorheben
        if highlight_nodes:
            for node in highlight_nodes:
                if node in nodes:
                    idx = list(nodes).index(node)
                    plt.scatter(values[idx], y_values[idx], color='red', edgecolor='black', s=150,
                                label=f'Hervorgehoben: {node}')

        plt.legend()
        plt.show()

    def get_norm_adj_mat(self, enable_sparse=False):
        self.is_sparse = is_sparse
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = self.user_num + self.item_num

        if enable_sparse:
            if not is_sparse:
                self.logger.warning(
                    "Import `torch_sparse` error, please install corrsponding version of `torch_sparse`. Now we will use dense edge_index instead of SparseTensor in dataset.")
            else:
                adj_t = self.edge_index_to_adj_t(edge_index, edge_weight, num_nodes, num_nodes)
                adj_t = gcn_norm(adj_t, None, num_nodes, add_self_loops=False)
                return adj_t, None

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=False)

        return edge_index, edge_weight

    def networkx_bipartite_graph(self):
        # build undirected and bipartite graph with networkx
        # print(f'{self.__class__.__name__}: building a bipartite graph with networkx')
        graph = networkx.Graph()
        # Get unique user and item indices
        user_indices = self.inter_feat[self.uid_field].unique()
        item_indices = self.inter_feat[self.iid_field].unique()

        user_indices = range(1, self.user_num)
        item_indices = range(self.user_num +1, self.user_num +  self.item_num)

        graph.add_nodes_from(user_indices, bipartite="users")
        graph.add_nodes_from(item_indices, bipartite="items")
        edges = [(u, i + self.user_num ) for u, i in
                 zip(self.inter_feat[self.uid_field].values, self.inter_feat[self.iid_field].values)]
        graph.add_edges_from(edges)

        if not networkx.is_connected(graph):
            print(f'{self.__class__.__name__}: the graph is not connected.')
        if not networkx.is_bipartite(graph):
            print(f'{self.__class__.__name__}: the graph is not bipartite.')

        return graph

    def connected_graph(self):
        graph = self.networkx_bipartite_graph()
        # if graph is not connected, retain only the biggest connected portion
        if not networkx.is_connected(graph):
            print(f'{self.__class__.__name__}: the graph is not connected. Building the connected subgraph.')

            # Find all connected components
            connected_components = list(networkx.connected_components(graph))

            # Identify the largest connected component
            largest_component = max(connected_components, key=len)

            # Find nodes in disconnected partitions
            disconnected_partitions = [component for component in connected_components if
                                       component != largest_component]

            graph = graph.subgraph(largest_component)

            print('The disconnected partitions: ', disconnected_partitions)

        print(f'{self.__class__.__name__}: graph is connected.')
        user_nodes, item_nodes = bipartite.sets(graph)
        print(f'{self.__class__.__name__}: graph characteristics'
              f'\n nodes: {len(graph.nodes)}'
              f'\n edges: {len(graph.edges)}'
              f'\n user nodes: {len(user_nodes)}'
              f'\n item nodes: {len(item_nodes)}')
        return graph

    def make_connected(self):
        self.bipartite_graph = self.connected_graph()

        user_nodes, item_nodes = bipartite.sets(self.bipartite_graph)
        item_nodes = {item - self.user_num for item in item_nodes}

        n_old_users = self.user_num
        n_old_items = self.item_num

        # filter out users and items
        inter_feat = self.inter_feat[
            self.inter_feat[self.uid_field].isin(user_nodes) & self.inter_feat[self.iid_field].isin(item_nodes)]
        #assert user_nodes == set(self.inter_feat[self.uid_field].unique()), f'{self.__class__.__name__}:' \
        #                                                                 f' a problem occurred during dataset filtering'
        #assert item_nodes == set(self.inter_feat[self.iid_field].unique()), f'{self.__class__.__name__}:' \
        #                                                                 f' a problem occurred during dataset filtering'

        print(f'{self.__class__.__name__}: {n_old_users - inter_feat[self.uid_field].nunique()} users removed')
        print(f'{self.__class__.__name__}: {n_old_items - inter_feat[self.iid_field].nunique()} items removed')

    def evaluate(self, best_user_nodes=None, worst_user_nodes=None):
        return {
            "inter_num": self.inter_num,
            "sparsity": self.sparsity,
            "connected": self.connected,
            "user_num": self.user_num,
            "item_num": self.item_num,
            "user_mean": self.avg_actions_of_users,
            "item_mean": self.avg_actions_of_items,
            "user_median": self.median_actions_of_users,
            "item_median": self.median_actions_of_items,
            "user_max": self.max_actions_of_users,
            "item_max": self.max_actions_of_items,
            "user_min": self.min_actions_of_users,
            "item_min": self.min_actions_of_items,
            "sparsity_log": self.sparsity_log,
            "space_size": self.space_size,
            "space_size_log": self.space_size_log,
            "shape": self.shape,
            "shape_log": self.shape_log,
            "gini_user": self.gini_user,
            "gini_item": self.gini_item,
            "average_degree": self.average_degree(),
            "average_degree_user": self.average_degree('user'),
            "average_degree_item": self.average_degree('item'),
            "average_degree_user_log": self.average_degree('user', True),
            "average_degree_item_log": self.average_degree('item', True),
            "degree_assort_user": self.degree_assortativity(self.user_nodes),
            "degree_assort_item": self.degree_assortativity(self.item_nodes),
            "average_clustering_coef_dot": self.calculate_average_clustering(method='dot',
                                                               nodes=self.bipartite_graph,
                                                               precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_dot_user": self.calculate_average_clustering(method='dot',
                                                                             nodes=self.user_nodes,
                                                                             precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_dot_item": self.calculate_average_clustering(method='dot',
                                                                             nodes=self.item_nodes,
                                                                             precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_min": self.calculate_average_clustering(method='min',
                                                                             nodes=self.bipartite_graph,
                                                                             precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_min_user": self.calculate_average_clustering(method='min',
                                                                                  nodes=self.user_nodes,
                                                                                  precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_min_item": self.calculate_average_clustering(method='min',
                                                                                  nodes=self.item_nodes,
                                                                                  precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_max": self.calculate_average_clustering(method='max',
                                                                             nodes=self.bipartite_graph,
                                                                             precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_max_user": self.calculate_average_clustering(method='max',
                                                                                  nodes=self.user_nodes,
                                                                                  precomputed_clustering=self.precomputed_clustering),
            "average_clustering_coef_max_item": self.calculate_average_clustering(method='max',
                                                                                  nodes=self.item_nodes,
                                                                                  precomputed_clustering=self.precomputed_clustering)
        }

    def evaluate_best_worst_users(self, best_user_nodes, worst_user_nodes):

        if not self.connected:
            existing_nodes = set(self.bipartite_graph.nodes)

            nodes_to_check = set(best_user_nodes)
            nodes_in_graph = nodes_to_check & existing_nodes
            degree_assort_best_users = self.degree_assortativity(nodes_in_graph)
            average_clustering_coef_dot_best_users = self.calculate_average_clustering(method='dot',
                                                               nodes=nodes_in_graph,
                                                               precomputed_clustering=self.precomputed_clustering)

            nodes_to_check = set(worst_user_nodes)
            nodes_in_graph = nodes_to_check & existing_nodes
            degree_assort_worst_users = self.degree_assortativity(nodes_in_graph)
            average_clustering_coef_dot_worst_users = self.calculate_average_clustering(method='dot',
                                                                nodes=nodes_in_graph,
                                                                precomputed_clustering=self.precomputed_clustering)
        else:
            degree_assort_best_users = self.degree_assortativity(best_user_nodes)
            degree_assort_worst_users = self.degree_assortativity(worst_user_nodes)
            average_clustering_coef_dot_best_users = self.calculate_average_clustering(method='dot',
                                                               nodes=best_user_nodes,
                                                               precomputed_clustering=self.precomputed_clustering)
            average_clustering_coef_dot_worst_users = self.calculate_average_clustering(method='dot',
                                                                nodes=worst_user_nodes,
                                                                precomputed_clustering=self.precomputed_clustering)


        return {
            "degree_assort_best_users": degree_assort_best_users,
            "degree_assort_worst_users": degree_assort_worst_users,
            "average_clustering_coef_dot_best_users": average_clustering_coef_dot_best_users,
            "average_clustering_coef_dot_worst_users": average_clustering_coef_dot_worst_users
        }