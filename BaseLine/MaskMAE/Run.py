import warnings
warnings.filterwarnings("ignore")

import os
import torch
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.decomposition import PCA


# Check CUDA availability
if torch.cuda.is_available():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)
    print("GPU Count:", torch.cuda.device_count())
    print("Current GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='imputation', key_added_pred='impute_mclust', random_seed=666):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs[key_added_pred] = mclust_res
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

from Func import graph_construction
from Utils import refine_label, getACC
from BaseLine.MaskMAE.Model.Models import MaskMAE

proj_name = '151671'
num_clusters = 5 if proj_name in ['151669', '151670', '151671', '151672'] else 7

with open('./Config/DLPFC.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

data_root = Path("/home/jinjie/JinJie/Mycode/DataAll/DLPFC")
count_file = proj_name + "_filtered_feature_bc_matrix.h5"
adata = sc.read_visium(data_root / proj_name, count_file=count_file)
adata.var_names_make_unique()

truth_path = "/home/jinjie/JinJie/Mycode/DataAll/DLPFC/Truth_Label/" + proj_name + '_truth.txt'
Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['layer_guess'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
adata = adata[~pd.isnull(adata.obs['layer_guess'])]
adata.layers['count'] = adata.X.toarray()

sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=config['top_genes'])
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
adata.obsm['X_pca'] = adata_X

graph_dict = graph_construction(adata, config['k_cutoff'])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = MaskMAE(adata, graph_dict=graph_dict, num_clusters=num_clusters, device=device, config=config)
net.trian()

recon = net.process()
recon = recon.data.cpu().numpy()
adata.obsm['latent'] = recon

adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='latent', key_added_pred='mclust')
adata.obs['domain'] = refine_label(adata, 30, key='mclust')

sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
ARI = ari_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
ACC = getACC(adata, 'layer_guess', 'domain')

file_path = f'result/ls/MaskMAE_{proj_name}_results.h5ad'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
adata.write(file_path)