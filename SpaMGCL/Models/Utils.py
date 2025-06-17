import numpy as np
import scanpy as sc
import pandas as pd
from scipy.spatial import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from scipy.spatial.distance import *


def res_search(adata, target_k=7, res_start=0.1, res_step=0.1, res_epochs=10):
    """
    Search for the optimal Leiden clustering resolution that yields the desired number of clusters.

    Args:
        adata (AnnData): Annotated data object.
        target_k (int): Target number of clusters.
        res_start (float): Starting resolution value.
        res_step (float): Step size for adjusting resolution.
        res_epochs (int): Maximum number of iterations.

    Returns:
        float: Recommended resolution value.
    """
    print(f"Searching resolution to get k={target_k}")
    res = res_start
    sc.tl.leiden(adata, resolution=res)

    old_k = len(adata.obs['leiden'].cat.categories)
    print("Res = ", res, "Num of clusters = ", old_k)

    run = 0
    while old_k != target_k:
        old_sign = 1 if (old_k < target_k) else -1
        sc.tl.leiden(adata, resolution=res + res_step * old_sign)
        new_k = len(adata.obs['leiden'].cat.categories)
        print("Res = ", res + res_step * old_sign, "Num of clusters = ", new_k)
        if new_k == target_k:
            res = res + res_step * old_sign
            print("Recommended res = ", str(res))
            return res
        new_sign = 1 if (new_k < target_k) else -1
        if new_sign == old_sign:
            res = res + res_step * old_sign
            print("Res changed to", res)
            old_k = new_k
        else:
            res_step = res_step / 2
            print("Res changed to", res)
        if run > res_epochs:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run += 1
    print("Recommended res = ", str(res))
    return res


def _compute_CHAOS(clusterlabel, location):
    """
    Compute the CHAOS score, which evaluates spatial coherence of clusters.

    Args:
        clusterlabel (array-like): Cluster labels for each cell.
        location (array-like): Spatial coordinates of cells.

    Returns:
        float: CHAOS score (lower means better spatial coherence).
    """
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count += 1

    return np.sum(dist_val) / len(clusterlabel)


def fx_1NN(i, location_in):
    """
    Helper function to compute the distance to the nearest neighbor within the same cluster.

    Args:
        i (int): Index of current point.
        location_in (np.ndarray): Array of spatial coordinates.

    Returns:
        float: Distance to the first nearest neighbor.
    """
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf  # Exclude self-distance
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):
    """
    Determine whether the majority of k-nearest neighbors belong to the same cluster.

    Args:
        i (int): Index of current point.
        location_in (np.ndarray): Spatial coordinates.
        k (int): Number of neighbors to consider.
        cluster_in (array-like): Cluster labels.

    Returns:
        int: 1 if majority of neighbors are from a different cluster, else 0.
    """
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


import ot


def refine_label(adata, radius=30, key='label'):
    """
    Refine cluster labels based on spatial proximity using majority voting among neighbors.

    Args:
        adata (AnnData): Annotated data object.
        radius (int): Number of nearest neighbors to consider.
        key (str): Key in `adata.obs` containing the original cluster labels.

    Returns:
        list: Refined cluster labels as strings.
    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # Extract spatial coordinates
    position = adata.obsm['spatial']
    # Compute pairwise Euclidean distances
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
    return new_type