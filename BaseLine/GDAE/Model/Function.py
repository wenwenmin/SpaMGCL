import warnings
from random import random

from torch import nn
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import SparseTensor

from MyGDAE.GDAE.Utilss import get_activation_layer

warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
import scanpy as sc
import pandas as pd


def generate_adj_mat_KNN(adata, include_self=False, n=6):
    from sklearn import metrics
    import numpy as np

    assert 'spatial' in adata.obsm

    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj_mat = np.zeros((len(adata), len(adata)))

    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n + 1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat


def generate_adj_mat_euclidean(adata, max_dist):
    from sklearn import metrics
    import numpy as np

    assert 'spatial' in adata.obsm

    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')

    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat


def graph_construction(adata, n=6, dmax=50, mode='KNN'):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat_KNN(adata, include_self=False, n=n)
    else:
        adj_m1 = generate_adj_mat_euclidean(adata, dmax)

    return adj_m1


def generate_topology_encoding(adj_matrix, encoding_size, device="cuda"):
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device=device)
    adj_sparse = SparseTensor.from_dense(adj_matrix)

    D_inv = (adj_sparse.sum(1).squeeze() + 1e-10) ** -1.0
    D_inv = D_inv.to(device)

    I = torch.eye(adj_matrix.shape[0], device=device)
    row, col = dense_to_sparse(I)[0]
    D_inv_sparse = SparseTensor(row=row, col=col, value=D_inv, sparse_sizes=(adj_matrix.shape[0], adj_matrix.shape[0]))

    P = adj_sparse @ D_inv_sparse
    M = P

    TE = [M.get_diag().float()]
    M_power = M
    M_power = M_power.half()
    M = M.half()

    for _ in tqdm(range(encoding_size - 1)):
        M_power = M_power @ M
        TE.append(M_power.get_diag().float())

    TE = torch.stack(TE, dim=-1)

    return TE