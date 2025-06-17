import os
from pathlib import Path
import warnings
import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.mixture import GaussianMixture

# 忽略警告
warnings.filterwarnings("ignore")

# 自定义模块（确保路径正确）
from Models import graph_construction, Mgac, refine_label

# mclust 聚类函数
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='imputation', key_added_pred='impute_mclust', random_seed=666):
    """使用 R 的 mclust 包进行聚类"""
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    from rpy2.robjects.numpy2ri import activate
    activate()
    robjects.r['set.seed'](random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2]).astype(int).astype('category')
    adata.obs[key_added_pred] = mclust_res
    return adata


# 设置设备（CPU/GPU）
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if device != 'cpu':
    print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")


# 加载配置文件
def load_config(config_path='./Config/DLPFC.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


# 数据预处理流程
def preprocess_data(data_root, proj_name, config):
    adata = sc.read_visium(data_root / proj_name, count_file=f"{proj_name}_filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()

    # 真实标签读取
    truth_path = data_root / f"{proj_name}_truth.txt"
    truth_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    truth_df.columns = ['Ground Truth']
    adata.obs['layer_guess'] = truth_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[~pd.isnull(adata.obs['layer_guess'])]

    # 构建图结构
    graph_dict = graph_construction(adata, config['k_cutoff'])

    # 数据预处理
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=config['top_genes'])
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    # PCA 降维
    pca = PCA(n_components=200, random_state=42)
    adata_X = pca.fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    return adata


# 训练 M-GAC 模型
def train_model(adata, graph_dict, num_clusters, config, device='cpu'):
    net = Mgac(adata, graph_dict=graph_dict, num_clusters=num_clusters, device=device, config=config)
    net.trian()
    enc_rep = net.process().data.cpu().numpy()
    adata.obsm['latent'] = enc_rep
    return adata


# 聚类与评估
def cluster_and_evaluate(adata, num_clusters):
    adata = mclust_R(adata, num_cluster=num_clusters, used_obsm='latent', key_added_pred='mclust')
    adata.obs['domain'] = refine_label(adata, 30, key='mclust')

    adata = adata[~pd.isnull(adata.obs['layer_guess'])]
    ari = ari_score(adata.obs['layer_guess'], adata.obs['domain'])
    return ari, adata


# 保存结果
def save_result(adata, output_dir, proj_name):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{proj_name}.h5ad")
    adata.write(output_path)
    print(f"结果已保存至: {output_path}")


# 单组织训练
def process_one_tissue(proj_name, data_root, config, output_dir, device='cpu'):
    try:
        print(f"\n---------- 正在处理项目: {proj_name} -----------")

        # 获取聚类数量
        num_clusters = 5 if proj_name in ['151669', '151670', '151671', '151672'] else 7

        # 数据预处理
        adata = preprocess_data(data_root, proj_name, config)

        # 构建图结构
        graph_dict = graph_construction(adata, config['k_cutoff'])

        # 模型训练
        adata = train_model(adata, graph_dict, num_clusters, config, device)

        # 聚类和评估
        ari, adata = cluster_and_evaluate(adata, num_clusters)

        # 保存结果
        save_result(adata, output_dir, proj_name)

        print(f"[{proj_name}] ARI: {ari:.4f}")
        return ari

    except Exception as e:
        print(f"处理失败: {proj_name}, 错误: {str(e)}")
        return None


# 主流程
def main():
    config = load_config()
    data_root = Path('/home/jinjie/JinJie/DataAll/DLPFC')
    output_dir = Path('RES')
    proj_names = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676"
    ]

    ari_list = []

    for proj_name in proj_names:
        ari = process_one_tissue(proj_name, data_root, config, output_dir, device)
        if ari is not None:
            ari_list.append(ari)

    if ari_list:
        avg_ari = sum(ari_list) / len(ari_list)
        mid_ari = sorted(ari_list)[len(ari_list) // 2]
        print("\n最终结果汇总:")

        print(f"平均 ARI: {avg_ari:.4f}")
        print(f"中位数 ARI: {mid_ari:.4f}")
    else:
        print("没有成功处理任何样本。")


if __name__ == '__main__':
    main()