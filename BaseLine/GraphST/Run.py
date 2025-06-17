import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
from pathlib import Path
from BaseLine.GraphST.Model.GraphST import GraphST
from BaseLine.GraphST.Model.GraphST import clustering

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

n_clusters = 7
dataset = '151676'
proj_name = dataset

data_root = Path('/home/jinjie/JinJie/JinJie/DataAll/DLPFC/')
adata = sc.read_visium(data_root / proj_name, count_file=proj_name + "_filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()

model = GraphST.GraphST(adata, device=device)
adata = model.train()

radius = 50
tool = 'mclust'

if tool == 'mclust':
    clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

truth_path = "/home/jinjie/JinJie/JinJie/DataAll/DLPFC/Truth_Label/" + proj_name + '_truth.txt'
Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['layer_guess'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

adata = adata[~pd.isnull(adata.obs['layer_guess'])]

ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['layer_guess'])
adata.uns['ARI'] = ARI

print('Dataset:', dataset)
print('ARI:', ARI)

def getACC(label_key, prediction_key):
    NMI = normalized_mutual_info_score(label_key, prediction_key)
    HOM = homogeneity_score(label_key, prediction_key)
    COM = completeness_score(label_key, prediction_key)
    return (NMI + HOM + COM) / 3

ACC = getACC(adata.obs['domain'], adata.obs['layer_guess'])
print('ACC:', ACC)

output_dir = Path('/home/jinjie/JinJie/JinJie/DataAll/DLPFC/GraphST')
output_dir.mkdir(parents=True, exist_ok=True)

adata.write(output_dir / f"{proj_name}_output.h5ad")