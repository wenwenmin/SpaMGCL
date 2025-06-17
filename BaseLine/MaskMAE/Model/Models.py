import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.optim import optimizer
from torch_geometric.nn import (
    Linear,  # 图卷积网络中的线性层
    GCNConv,  # 图卷积（GCN）层
    SAGEConv,  # 图注意力网络（GraphSAGE）层
    GATConv,  # 图注意力网络（GAT）层
    GINConv,  # 图同态网络（GIN）层
    GATv2Conv,  # 图注意力网络（GATv2）层
    global_add_pool,  # 全局加池化操作
    global_mean_pool,  # 全局均值池化操作
    global_max_pool, TransformerConv  # 全局最大池化操作
)

from torch_geometric.utils import add_self_loops, negative_sampling

import torch
import torch.nn as nn
from torch_sparse import SparseTensor


def create_gnn_layer(in_channels, out_channels):
    return GCNConv(in_channels, out_channels)


def create_activation_layer(name):
    activations = {"relu": nn.ReLU, "elu": nn.ELU, None: nn.Identity}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, bn=False,
                 activation="elu", num_nodes=None, node_emb=None):
        super().__init__()
        if node_emb and not num_nodes:
            raise ValueError("`num_nodes` must be specified when `node_emb` is used.")
        # GNN layers
        self.conv1 = create_gnn_layer(in_channels, hidden_channels)
        self.conv2 = create_gnn_layer(hidden_channels, out_channels)

        # Batch norm, dropout, activation
        self.bn1 = nn.BatchNorm1d(hidden_channels) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation_layer(activation)

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.bn1, self.bn2]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, edge_index):
        x = self.activation(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.conv2(x, edge_index)))
        return x



class DotEdgeDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)
        if sigmoid:
            return x.sigmoid()
        else:
            return x



class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.5, activation='relu'):
        super(EdgeDecoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation_layer(activation)

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if sigmoid:
            return x.sigmoid()
        else:
            return x


import torch
import torch.nn as nn

class DegreeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, activation='relu'):
        super(DegreeDecoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation_layer(activation)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg=1):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()

import torch
from torch import Tensor

def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges


from torch_geometric.utils import to_undirected, sort_edge_index, degree
import torch

def mask_edge(edge_index: torch.Tensor, p: float=0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 (got {p})')
    edge_index = torch.stack(edge_index, dim=0)
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]

def adj_matrix_to_edges(adj_matrix):
    if adj_matrix.is_sparse:
        adj_matrix = adj_matrix.to_dense()
    edge_index = adj_matrix.nonzero(as_tuple=True)
    return edge_index

class Jin(nn.Module):
    def __init__(self, features_dims, config, bn=False, att_dropout_rate=0.2, use_token=True):
        super(Jin, self).__init__()
        self.config = config
        [input_dim, hidden_dim, latent_dim, output_dim, num] = features_dims
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.edge_decoder = EdgeDecoder(latent_dim, hidden_dim)
        self.degree_decoder = DegreeDecoder(in_channels=latent_dim, hidden_channels=hidden_dim, out_channels=num)
        self.negative_sampler = random_negative_sampler

    def forward(self, feature, edge_index):
        x = feature
        edge_index = edge_index
        edge_index = adj_matrix_to_edges(edge_index)
        remaining_edges, masked_edges = mask_edge(edge_index)
        edge_index = torch.stack(edge_index, dim=0)
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=x.size(0),
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)
        z = self.encoder(x, remaining_edges)
        perm = torch.randperm(masked_edges.size(1))
        batch_masked_edges = masked_edges[:, perm]
        batch_neg_edges = neg_edges[:, perm]
        pos_out = self.edge_decoder(z, batch_masked_edges, sigmoid=False)
        neg_out = self.edge_decoder(z, batch_neg_edges, sigmoid=False)
        loss = info_nce_loss(pos_out, neg_out)
        deg = degree(masked_edges[1].flatten(), num_nodes=x.size(0)).float()
        degs = F.mse_loss(self.degree_decoder(z).squeeze(), deg)
        degs = 0
        return loss, degs

    @torch.no_grad()
    def evaluate(self, datas, edges):
        enc_rep = self.encoder(datas, edges)
        return enc_rep