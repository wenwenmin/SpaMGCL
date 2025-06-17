import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score, completeness_score,
                             fowlkes_mallows_score, v_measure_score, homogeneity_score)
from sklearn.metrics import *

import ot  # 这里假设你已经安装了OT库来计算距离矩阵

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv




def get_gnn_layer(name, in_channels, out_channels, heads):
    if name == "sage":
        layer = SAGEConv(in_channels, out_channels)
    elif name == "gcn":
        layer = GCNConv(in_channels, out_channels)
    elif name == "gin":
        layer = GINConv(Linear(in_channels, out_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, out_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, out_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer


def get_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unknown activation")


def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def topology_recon_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def refine_label(adata, radius=30, key='label'):
    n_neigh = radius
    old_type = adata.obs[key].values
    new_type = []
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])

        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
    new_type = [str(i) for i in list(new_type)]
    adata.obs['label_refined'] = np.array(new_type)

    return new_type

def getACC(adata, label_key, prediction_key):
    adata = adata[~pd.isnull(adata.obs[label_key])]
    NMI = compute_NMI(adata, label_key, prediction_key)
    HOM = compute_HOM(adata, label_key, prediction_key)
    COM = compute_COM(adata, label_key, prediction_key)
    return (NMI + HOM + COM)/3

def compute_ARI(adata, gt_key, pred_key):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_NMI(adata, gt_key, pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_HOM(adata, gt_key, pred_key):
    return homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])


def compute_COM(adata, gt_key, pred_key):
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])



def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)

def topology_recon_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss