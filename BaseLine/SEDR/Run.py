import scanpy as sc
import pandas as pd
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
import torch
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import SEDR
import numpy as np


def getACC(label_key, prediction_key):
    NMI = normalized_mutual_info_score(label_key, prediction_key)
    HOM = homogeneity_score(label_key, prediction_key)
    COM = completeness_score(label_key, prediction_key)
    return (NMI + HOM + COM) / 3


def mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR', random_seed=2023):
    modelNames = 'EEE'
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[use_rep]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


def train_one_slice(sample_name):
    random_seed = 0
    SEDR.fix_seed(random_seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_root = Path("/home/jinjie/JinJie/DataAll/ME9_5")
    save_dir = data_root / "SEDR"
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create directory: {save_dir}. Please check permissions.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Base path {data_root} does not exist. Please verify the path.")
    data_file = data_root / "E9.5_E1S1.MOSTA.h5ad"
    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {data_file}")
    adata = sc.read_h5ad(data_file)
    adata.var_names_make_unique()
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    from sklearn.decomposition import PCA
    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    graph_dict = SEDR.graph_construction(adata, 10)
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR_feat'] = sedr_feat
    sedr_recon = sedr_net.recon()
    adata.obsm['SEDR_recon'] = sedr_recon
    n_clusters = 7
    adata = mclust_R(adata, n_clusters, use_rep='SEDR_feat', key_added='mclust')
    save_path = save_dir / f"{sample_name}_SEDR.h5ad"
    adata.write_h5ad(save_path)
    print(f"Saved h5ad file to: {save_path}")
    return adata


def train_me9_5():
    sample_name = "E9.5_E1S1"
    adata = train_one_slice(sample_name)
    print("训练完成。")


if __name__ == '__main__':
    train_me9_5()