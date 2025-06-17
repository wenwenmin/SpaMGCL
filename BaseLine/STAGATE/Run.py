import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.cluster import adjusted_rand_score

# 自定义模块
import STAGATE_pyG
from BaseLine.STAGATE.STAGATE_pyG.utils import getACC


# 设置随机种子（如需）
def set_random_seed(seed=42):
    np.random.seed(seed)


# 加载数据并进行预处理
def load_and_preprocess_data(data_root, sample_name):
    count_file = f"{sample_name}_filtered_feature_bc_matrix.h5"
    adata = sc.read_visium(data_root / sample_name, count_file=count_file)
    adata.var_names_make_unique()

    # HVG + normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


# 构建空间网络
def build_spatial_network(adata, radius=150):
    STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=radius)
    STAGATE_pyG.Stats_Spatial_Net(adata)
    return adata


# 训练 STAGATE 模型
def train_stagate_model(adata):
    adata = STAGATE_pyG.train_STAGATE(adata)
    sc.pp.neighbors(adata, use_rep='STAGATE')
    sc.tl.umap(adata)
    return adata


# 聚类与评估
def cluster_and_evaluate(adata, n_clusters, truth_path):
    adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)

    # 读取真实标签
    truth_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    truth_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = truth_df.loc[adata.obs_names, 'Ground Truth']

    # 过滤空值
    valid_adata = adata[~pd.isnull(adata.obs['Ground Truth'])]

    # 评估指标
    ari = adjusted_rand_score(valid_adata.obs['Ground Truth'], valid_adata.obs['mclust'])
    acc = getACC(valid_adata.obs['Ground Truth'], valid_adata.obs['mclust'])

    return ari, acc, adata


# 保存结果
def save_result(adata, output_dir, sample_name):
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "STAGATE_pyG.h5ad"
    adata.write(str(output_path))
    print(f"Saved result to {output_path}")


# 单个样本处理流程
def train_one_slice(sample_name, config):
    try:
        data_root = config['data_root']
        truth_dir = config['truth_dir']
        output_base = config['output_dir']
        n_clusters = config['n_clusters_map'].get(sample_name, 7)

        # Step 1: 数据加载 & 预处理
        adata = load_and_preprocess_data(data_root, sample_name)

        # Step 2: 空间网络构建
        build_spatial_network(adata, radius=config.get('rad_cutoff', 150))

        # Step 3: 模型训练
        train_stagate_model(adata)

        # Step 4: 真实标签 & 聚类评估
        truth_path = os.path.join(truth_dir, f"{sample_name}_truth.txt")
        ari, acc, adata = cluster_and_evaluate(adata, n_clusters, truth_path)

        # Step 5: 保存结果
        output_dir = Path(output_base) / sample_name
        save_result(adata, output_dir, sample_name)

        print(f"[{sample_name}] ARI: {ari:.4f}, ACC: {acc:.4f}")
        return ari, acc

    except Exception as e:
        print(f"Error processing {sample_name}: {e}")
        return 0.0, 0.0


# 主函数：批量处理多个组织切片
def main():
    config = {
        'data_root': Path("/home/jinjie/JinJie/DataAll/DLPFC"),
        'truth_dir': "/home/jinjie/JinJie/DataAll/DLPFC/Truth_Label",
        'output_dir': "/home/jinjie/JinJie/DataAll/DLPFC/STAGATE_pyG",
        'rad_cutoff': 150,
        'n_clusters_map': {
            '151669': 5, '151670': 5, '151671': 5, '151672': 5,
            '151507': 7, '151508': 7, '151509': 7, '151510': 7,
            '151673': 7, '151674': 7, '151675': 7, '151676': 7
        }
    }

    section_ids = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676"
    ]

    ari_list, acc_list = [], []

    for tissue in section_ids:
        ari, acc = train_one_slice(tissue, config)
        ari_list.append(ari)
        acc_list.append(acc)

    median_ari = np.median(ari_list)
    median_acc = np.median(acc_list)

    print("\nMedian Scores Across All Tissues:")
    print(f"ARI: {median_ari:.4f}, ACC: {median_acc:.4f}")


if __name__ == "__main__":
    set_random_seed()
    main()