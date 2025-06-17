#
# import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, block_diag

##### Generate adjacency matrix
def generate_adj_mat(adata, include_self=False, n=6):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should contain spatial information'

    # Compute pairwise distances between spatial locations
    dist = metrics.pairwise_distances(adata.obsm['spatial'])

    # The commented code below constructs adjacency based on distance threshold
    # sample_name = list(adata.uns['spatial'].keys())[0]
    # scalefactors = adata.uns['spatial'][sample_name]['scalefactors']
    # adj_mat = dist <= scalefactors['fiducial_diameter_fullres'] * (n+0.2)
    # adj_mat = adj_mat.astype(int)

    # Build adjacency matrix using k nearest neighbors
    adj_mat = np.zeros((len(adata), len(adata)))  # Initialize adjacency matrix with zeros
    for i in range(len(adata)):
        # Find the n nearest neighbors (including itself) for each point
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1  # Connect current point to its n nearest neighbors

    # If not including self-loop, set diagonal elements to 0
    if not include_self:
        x, y = np.diag_indices_from(adj_mat)  # Get indices of diagonal elements
        adj_mat[x, y] = 0  # Set diagonal values to 0 to exclude self

    # Create symmetric adjacency matrix
    adj_mat = adj_mat + adj_mat.T  # Make matrix symmetric
    adj_mat = adj_mat > 0  # Convert values to boolean: 1 means connected, 0 otherwise
    adj_mat = adj_mat.astype(np.int64)  # Convert to integer type

    return adj_mat  # Return adjacency matrix


def generate_adj_mat_1(adata, max_dist):
    from sklearn import metrics
    assert 'spatial' in adata.obsm, 'AnnData object should contain spatial information'

    # Compute pairwise Euclidean distances between spatial positions
    dist = metrics.pairwise_distances(adata.obsm['spatial'], metric='euclidean')

    # Generate adjacency matrix based on maximum distance threshold
    adj_mat = dist < max_dist  # Connect if distance is less than max distance
    adj_mat = adj_mat.astype(np.int64)  # Convert to integer type, True becomes 1, False becomes 0
    return adj_mat  # Return adjacency matrix


##### Convert sparse matrix to Torch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a sparse matrix to a PyTorch sparse tensor."""
    # Convert sparse matrix to COO format and cast to float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # Extract row and column indices
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # Extract values
    values = torch.from_numpy(sparse_mx.data)
    # Get shape of sparse matrix
    shape = torch.Size(sparse_mx.shape)
    # Return PyTorch sparse tensor
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_graph(adj):
    # Add identity matrix to adjacency matrix to allow self-connections
    adj_ = adj + sp.eye(adj.shape[0])
    # Calculate degree (number of connections per node)
    rowsum = np.array(adj_.sum(1))
    # Calculate inverse square root of degree matrix
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # Normalize adjacency matrix (degree normalization)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # Return normalized adjacency matrix as a sparse tensor
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def mask_generator(adj_label, N=1):
    # Get indices from adjacency label matrix
    idx = adj_label.indices()
    # Get total number of cells
    cell_num = adj_label.size()[0]

    # List to store non-neighbors
    list_non_neighbor = []
    for i in range(0, cell_num):
        # Get neighbors of node i
        neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
        # Number of selected non-neighbors (multiplied by N)
        n_selected = len(neighbor) * N

        # Get indices of non-neighbors
        total_idx = torch.range(0, cell_num-1, dtype=torch.float32)
        non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
        # Randomly select non-neighbors
        indices = torch.randperm(len(non_neighbor), dtype=torch.float32)
        random_non_neighbor = indices[:n_selected]
        list_non_neighbor.append(random_non_neighbor)

    # Merge indices of adjacency matrix with selected non-neighbors
    x = adj_label.indices()[0]
    y = torch.concat(list_non_neighbor)

    # Concatenate indices
    indices = torch.stack([x, y])
    indices = torch.concat([adj_label.indices(), indices], axis=1)

    # Concatenate values from adjacency matrix and new mask values (0 indicates no connection)
    value = torch.concat([adj_label.values(), torch.zeros(len(x), dtype=torch.float32)])
    # Generate final adjacency mask
    adj_mask = torch.sparse_coo_tensor(indices, value)

    return adj_mask


def graph_computing(pos, n):
    from scipy.spatial import distance
    # List to store adjacency matrix data
    list_x = []
    list_y = []
    list_value = []

    # Iterate through each node and compute distances to others
    for node_idx in range(len(pos)):
        tmp = pos[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, pos, 'euclidean')
        res = distMat.argsort()  # Sort by distance

        # Get top n nearest neighbors and create edges
        for j in np.arange(1, n + 1):
            list_x += [node_idx, res[0][j]]
            list_y += [res[0][j], node_idx]
            list_value += [1, 1]  # Value of 1 indicates a connection

    # Construct sparse matrix
    adj = sp.csr_matrix((list_value, (list_x, list_y)))
    # Set values >=1 to True, indicating a connection
    adj = adj >= 1
    adj = adj.astype(np.float32)
    return adj

def graph_construction(adata, n=6, dmax=50, mode='KNN'):
    """
    Construct adjacency matrix for graph
    Parameters:
    adata -- AnnData object containing spatial data
    n -- Number of neighbors, used in KNN mode
    dmax -- Maximum distance, used in distance mode
    mode -- Mode, 'KNN' uses KNN method, other values use distance-based
    Returns:
    graph_dict -- Dictionary containing normalized adjacency matrix and other graph info
    """
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n)
        # adj_m1 = graph_computing(adata.obsm['spatial'], n=n)  # Optional alternative method
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax)  # Use distance to generate adjacency matrix

    adj_m1 = sp.coo_matrix(adj_m1)  # Convert adjacency matrix to COO format

    # Remove self-loops (diagonal entries)
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()  # Remove zero entries

    # Preprocess graph (normalize)
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_m1 = adj_m1 + sp.eye(adj_m1.shape[0])  # Add self-loop to each node
    # adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())  # Optional label matrix

    adj_m1 = adj_m1.tocoo()  # Convert to COO format
    shape = adj_m1.shape
    values = adj_m1.data
    indices = np.stack([adj_m1.row, adj_m1.col])  # Get indices and values
    adj_label_m1 = torch.sparse_coo_tensor(indices, values, shape)  # Convert to sparse tensor

    # Compute normalization constant
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    # # Generate random mask (optional)
    # adj_mask = mask_generator(adj_label_m1.to_sparse(), N)

    graph_dict = {
        "adj_norm": adj_norm_m1,  # Normalized adjacency matrix
        "adj_label": adj_label_m1.coalesce(),  # Label matrix, ensure sparse format
        "norm_value": norm_m1,  # Normalization value
        # "mask": adj_mask  # Optional mask
    }

    return graph_dict

def coo2csr(coo_matrix):
    """
    Convert a COO-format sparse matrix to CSR format
    Parameters:
    coo_matrix -- Sparse matrix in COO format
    Returns:
    csr_matrix -- Sparse matrix in CSR format
    """
    coo_matrix = coo_matrix.coalesce()  # Ensure the sparse matrix is coalesced
    indices = coo_matrix.indices()  # Get indices
    values = coo_matrix.values()  # Get values
    sparse_matrix = sp.coo_matrix((values.numpy(), indices.numpy()), shape=coo_matrix.size())  # Create COO matrix
    csr_matrix = sparse_matrix.tocsr()  # Convert to CSR format
    return csr_matrix

def csr2coo(csr_matrix):
    """
    Convert a CSR-format sparse matrix to COO format
    Parameters:
    csr_matrix -- Sparse matrix in CSR format
    Returns:
    sparse_tensor -- Sparse tensor in COO format
    """
    coo_matrix = csr_matrix.tocoo()  # Convert CSR to COO
    indices = torch.tensor([coo_matrix.row, coo_matrix.col])  # Get row and column indices
    values = torch.tensor(coo_matrix.data)  # Get non-zero values
    size = torch.Size(coo_matrix.shape)  # Get matrix dimensions
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size)  # Create sparse tensor
    return sparse_tensor

def combine_graph_dict_1(dict_1, dict_2):
    """
    Combine two graph dictionaries
    Parameters:
    dict_1 -- First graph dictionary
    dict_2 -- Second graph dictionary
    Returns:
    graph_dict -- Combined graph dictionary
    """
    # Combine normalized adjacency matrices into one large graph
    tmp_adj_norm = csr2coo(block_diag([coo2csr(dict_1['adj_norm']), coo2csr(dict_2['adj_norm'])]))
    tmp_adj_label = csr2coo(block_diag([coo2csr(dict_1['adj_label']), coo2csr(dict_2['adj_label'])]))
    graph_dict = {
        "adj_norm": tmp_adj_norm,  # Combined normalized adjacency matrix
        "adj_label": tmp_adj_label,  # Combined label matrix
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])  # Average normalization value
    }
    return graph_dict

def combine_graph_dict(dict_1, dict_2):
    """
    Combine two graph dictionaries (using PyTorch tensors)
    Parameters:
    dict_1 -- First graph dictionary
    dict_2 -- Second graph dictionary
    Returns:
    graph_dict -- Combined graph dictionary
    """
    # Combine normalized adjacency matrices into one large graph
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())  # Combine normalized adjacency matrices
    tmp_adj_norm = tmp_adj_norm.to_sparse()  # Convert back to sparse matrix
    tmp_adj_label = torch.block_diag(dict_1['adj_label'].to_dense(), dict_2['adj_label'].to_dense())  # Combine label matrices
    tmp_adj_label = tmp_adj_label.to_sparse()  # Convert back to sparse matrix
    graph_dict = {
        "adj_norm": tmp_adj_norm,  # Combined normalized adjacency matrix
        "adj_label": tmp_adj_label,  # Combined label matrix
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])  # Average normalization value
    }
    return graph_dict