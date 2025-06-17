import warnings
from BaseLine.GraphMAE.Model.Models import graph_construction, refine_label, Mgac

warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import os
import yaml
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = '/home/jinjie/miniconda3/envs/SpaMGCL/lib/R'
os.environ['R_USER'] = '/home/jinjie/miniconda3/envs/SpaMGCL/lib/python3.10/site-packages/rpy2'

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

print("PyTorch 版本:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("GPU 设备数量:", torch.cuda.device_count())
    print("当前 GPU 设备名称:", torch.cuda.get_device_name(0))
else:
    print("CUDA 不可用")

proj_name = '151509'
num_clusters = 5 if proj_name in ['151669', '151670', '151671', '151672'] else 7

with open('./Config/DLPFC.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

data_root = Path('/home/jinjie/JinJie/SpaCMGAE/SpaCMGAE/DataAll/DLPFC') / proj_name
adata = sc.read_visium(data_root, count_file=f"{proj_name}_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()

file_path = data_root / f"{proj_name}_truth.txt"
Ann_df = pd.read_csv(file_path, sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['layer_guess'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
adata = adata[~pd.isnull(adata.obs['layer_guess'])]

graph_dict = graph_construction(adata, config['k_cutoff'])

print('---------- Graph construction finished! ----------')
print(graph_dict)

adata.layers['count'] = adata.X.toarray()
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=config['top_genes'])
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

from sklearn.decomposition import PCA
adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
adata.obsm['X_pca'] = adata_X
print('---------- Data Precede finished! ----------')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = GraphMAE(adata, graph_dict=graph_dict, num_clusters=num_clusters, device=device, config=config)
net.trian()

enc_rep = net.process()
enc_rep = enc_rep.data.cpu().numpy()
adata.obsm['latent'] = enc_rep

adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='latent', key_added_pred='mclust')
adata.obs['domain'] = refine_label(adata, 30, key='mclust')
sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]

from sklearn.metrics import adjusted_rand_score as ari_score
ARI = ari_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
print(ARI)

file_path = f'result/9SapaCMAGE_{proj_name}_results.h5ad'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
adata.write(file_path)
