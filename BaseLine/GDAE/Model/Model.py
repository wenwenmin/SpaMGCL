import numpy as np
from GDAE.Utilss import  get_gnn_layer, get_activation_layer,topology_recon_loss, to_sparse_tensor
import torch
from torch_geometric.utils import negative_sampling, add_self_loops, to_undirected
from tqdm import tqdm

try:
    import torch_cluster
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None


from GDAE.Utilss import *

import torch.nn as nn
from torch_geometric.nn import GCNConv


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, topo_size, hidden_channels, out_channels, dropout=0.5, norm=False, activation="tanh"):
        super().__init__()

        self.activation = get_activation_layer(activation)
        bn = nn.BatchNorm1d if norm else nn.Identity

        self.mlpFeature = nn.ModuleList([
            nn.Linear(input_dim, hidden_channels),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        ])
        self.bnFeature1 = bn(hidden_channels)
        self.bnFeature2 = bn(out_channels)

        self.mlpTopo = nn.ModuleList([
            nn.Linear(topo_size, hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        ])
        self.bnTopo = bn(hidden_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature, topo):
        feature = self.activation(self.mlpFeature[0](feature))
        topo = self.activation(self.mlpTopo[0](topo))
        combined_input = torch.cat([feature, topo], dim=1)

        feature = self.activation(self.bnFeature1(self.mlpFeature[1](combined_input)))
        feature = self.dropout(feature)
        topo = self.dropout(topo)

        feature = self.bnFeature2(self.mlpFeature[2](feature))
        topo = self.bnTopo(topo)

        return feature, topo


class MPGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, norm=False, activation="relu"):
        super(MPGNNEncoder, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        bn = nn.BatchNorm1d if norm else nn.Identity

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels

            self.convs.append(GCNConv(first_channels, second_channels))
            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)

        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)

        return x


class NodeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, activation='relu'):
        super(NodeDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels)
        )

        if activation is None:
            self.activation = None
        else:
            self.activation = get_activation_layer(activation)

    def forward(self, x):
        decoding = self.mlp(x)
        if self.activation is not None:
            decoding = self.activation(decoding)
        return decoding

    def sce_loss(self, x, y, alpha=2):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        loss = (1 - (x * y).sum(dim=1)).pow_(alpha)
        return loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.5, activation='relu'):
        super().__init__()

        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, z2, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x2 = z2[edge[0]] * z2[edge[1]]

        x = torch.cat([x, x2], dim=1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.5, activation='relu'):
        super(DegreeDecoder, self).__init__()

        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_layer(activation)

    def forward(self, x):
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)
        return x


class Masker(nn.Module):
    def __init__(self, p: float = 0.7, undirected: bool = True, walks_per_node: int = 1, walk_length: int = 3):
        super(Masker, self).__init__()
        self.p = p
        self.undirected = undirected
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length

    def random_walk_remove_edges(self, adj_matrix: np.ndarray, remove_prob: float = 0.3):
        num_nodes = adj_matrix.shape[0]
        mask_matrix = np.random.rand(num_nodes, num_nodes) < (1 - remove_prob)
        np.fill_diagonal(mask_matrix, 0)
        mask_matrix = mask_matrix.astype(bool)
        mask_matrix = mask_matrix * adj_matrix

        remain_matrix = adj_matrix - mask_matrix

        mask_tensor = torch.tensor(mask_matrix, dtype=torch.int64)
        remain_tensor = torch.tensor(remain_matrix, dtype=torch.int64)

        return mask_tensor, remain_tensor

    def forward(self, adj_matrix: np.ndarray):
        mask_matrix, remain_matrix = self.random_walk_remove_edges(adj_matrix, remove_prob=self.p)
        return mask_matrix, remain_matrix




class GDAES(nn.Module):
    def __init__(self, num_clusters, adj_dim, tisset_dim, topology_dim, input_dim, config,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.num_clusters = num_clusters
        self.input_dim = input_dim
        self.device = device
        self.adj_dim = adj_dim
        self.topology_dim = topology_dim
        self.tisset_dim = tisset_dim

        self.p = config['p']
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        self.encoder_activation = config['encoder_activation']
        self.encoder_dropout = config['encoder_dropout']
        self.decoder_dropout = config['decoder_dropout']
        self.encoder_channels = config['encoder_channels']
        self.hidden_channels = config['hidden_channels']
        self.decoder_channels = config['decoder_channels']
        self.top_loss = topology_recon_loss

        self.featyerDim = config['featyerDim']
        self.topoDim = config['topoDim']

        self.Masker = Masker(p=self.p)
        self.MLP_ENCODER = MLPEncoder(input_dim=self.featyerDim, topo_size=self.topoDim,
                                      hidden_channels=self.encoder_channels, out_channels=self.hidden_channels,
                                      dropout=self.encoder_dropout)
        self.MPGNN_ENCODER = MPGNNEncoder(in_channels=self.encoder_channels,
                                          hidden_channels=self.encoder_channels,
                                          out_channels=self.hidden_channels)
        self.EDGE_DECODER = EdgeDecoder(in_channels=self.featyerDim + self.hidden_channels,
                                        hidden_channels=self.decoder_channels)
        self.NODE_DECODER = NodeDecoder(in_channels=self.hidden_channels,
                                        hidden_channels=self.decoder_channels,
                                        out_channels=self.featyerDim + self.topoDim)
        self.DEGREE_DECODER = DegreeDecoder(in_channels=self.hidden_channels * 2,
                                            hidden_channels=self.decoder_channels * 4,
                                            out_channels=1)

    def reset_parameters(self):
        self.Masker.reset_parameters()
        self.MLP_ENCODER.reset_parameters()
        self.MPGNN_ENCODER.reset_parameters()
        self.EDGE_DECODER.reset_parameters()
        self.NODE_DECODER.reset_parameters()
        self.DEGREE_DECODER.reset_parameters()

    def forward(self, X, adj_matrix, topology, optimizer):
        return self.LogicalGDAE(X, adj_matrix, topology, optimizer)

    def LogicalGDAE(self, X, adj_matrix, topology, optimizer, epochs=50, grad_norm=1.0, lam1=0.1, lam2=0.001):
        total_loss = 0.0

        for epoch in tqdm(range(epochs), desc="训练进度", unit="轮"):
            feature = X
            adj_matrix = adj_matrix
            topology = topology

            remaining_edges, masked_edges = self.Masker(adj_matrix)

            loss_total = 0.0
            edge_index = torch.tensor(np.nonzero(adj_matrix), dtype=torch.long)
            edge_index = to_undirected(edge_index)
            edge_index, _ = add_self_loops(edge_index)

            num_nodes = adj_matrix.shape[0]
            num_neg_samples = masked_edges.size(1)
            neg_edges = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)
            neg_edges = neg_edges.view(2, -1)

            if feature.shape[0] == topology.shape[0]:
                x0 = torch.cat([feature, topology], dim=1)
            else:
                print(f"Error: feature and topology have different numbers of nodes.")
                continue

            optimizer.zero_grad()
            remaining_edges = remaining_edges.nonzero().T.to(self.device)
            feature = feature.to(self.device)

            z = self.MPGNN_ENCODER(feature, remaining_edges)
            z2, p = self.MLP_ENCODER(feature, topology)

            batch_masked_edges = masked_edges
            batch_neg_edges = neg_edges

            pos_out1 = self.EDGE_DECODER(z, p, batch_masked_edges, sigmoid=False)
            neg_out1 = self.EDGE_DECODER(z, p, batch_neg_edges, sigmoid=False)
            loss_edge = self.top_loss(pos_out1, neg_out1)

            decoding = self.NODE_DECODER(z2)
            loss_feature = self.NODE_DECODER.sce_loss(x0, decoding)

            degree_pred = self.DEGREE_DECODER(torch.cat([z, z2], dim=1))

            adj_matrix_torch = torch.from_numpy(adj_matrix).float().to(self.device)
            degree_target = adj_matrix_torch.sum(dim=1)

            loss_degree = F.mse_loss(degree_pred, degree_target)

            loss = loss_edge + lam1 * loss_feature + lam2 * loss_degree

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / epochs
        print(f"训练完成，平均损失：{avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, x, adj):
        adj = torch.tensor(adj)
        adj = adj.nonzero().T.to(self.device)
        x = x.to(self.device)
        enc_rep = self.MPGNN_ENCODER(x, adj)
        return enc_rep