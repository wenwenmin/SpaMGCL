import os
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
import scanpy as sc
import yaml
from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.decomposition import PCA

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['R_USER'] = '/home/jinjie/miniconda3/envs/Po/lib/python3.10/site-packages/rpy2'

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='imputation', key_added_pred='impute_mclust', random_seed=666):
    """Clustering using the mclust algorithm."""
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


# Print PyTorch and CUDA info
print("PyTorch 版本:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("GPU 设备数量:", torch.cuda.device_count())
    print("当前 GPU 设备名称:", torch.cuda.get_device_name(0))
else:
    print("CUDA 不可用")


# Project settings
proj_name = '151508'
num_clusters = 5 if proj_name in ['151669', '151670', '151671', '151672'] else 7

# Load config
with open('./Config/My.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

# Import custom modules
from GDAE.Function import generate_topology_encoding, graph_construction

# Data loading and preprocessing
data_root = Path("/home/jinjie/JinJie/Mycode/DataAll/DLPFC")
count_file = proj_name + "_filtered_feature_bc_matrix.h5"
adata = sc.read_visium(data_root / proj_name, count_file=count_file)
adata.var_names_make_unique()

truth_path = "/home/jinjie/JinJie/Mycode/DataAll/DLPFC/Truth_Label/" + proj_name + '_truth.txt'
Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['layer_guess'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
adata = adata[~pd.isnull(adata.obs['layer_guess'])]

# Graph construction
graph_dict = graph_construction(adata, dmax=config['data']['En'], mode=config['data']['KE'])
graph_topology = generate_topology_encoding(graph_dict, config['data']['step'])

# Preprocessing
adata.layers['count'] = adata.X.toarray()
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.filter_genes(adata, min_counts=10)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=config['data']['top_genes'])
adata = adata[:, adata.var['highly_variable'] == True]
sc.pp.scale(adata)

adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
adata.obsm['X_pca'] = adata_X

# Model training
from MyGDAE.GDAE.CDAE import Cdae

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = Cdae(adata, graph_dict=graph_dict, graph_topology=graph_topology, num_clusters=num_clusters, device=device, config=config)

net.trian()

# Inference
recon = net.process()
recon = recon.data.cpu().numpy()
adata.obsm['latent'] = recon
print(recon.shape)

# Clustering and evaluation
from MyGDAE.GDAE import Utilss

adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='latent', key_added_pred='mclust')
adata.obs['domain'] = Utilss.refine_label(adata, 30, key='mclust')

sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
ARI = ari_score(sub_adata.obs['layer_guess'], sub_adata.obs['domain'])
ACC = Utilss.getACC(adata, 'layer_guess', 'domain')
print('ARI:', ARI)
print('ACC:', ACC)

plot_color = ['#96EE86', '#F9F871', '#FFC75F', '#FF9671', '#FF6F91', '#D65DB1', '#845EC2']

ACC = Utilss.getACC(adata, 'layer_guess', 'domain')
print(ARI, ACC)

# Save results
directory = 'Result'
if not os.path.exists(directory):
    os.makedirs(directory)

file_path = os.path.join(directory, f'{proj_name}.adata')
adata.write(file_path)

print(f"结果已保存至：{file_path}")