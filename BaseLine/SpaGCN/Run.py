import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import cv2
import random
import torch
import warnings
warnings.filterwarnings("ignore")

# 自定义模块
from BaseLine.SpaGCN.Model import SpaGCN as spg
from sklearn.metrics import (
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    adjusted_rand_score,
    v_measure_score,
)

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ACC 计算函数
def getACC(label_true, label_pred):
    nmi = normalized_mutual_info_score(label_true, label_pred)
    hom = homogeneity_score(label_true, label_pred)
    com = completeness_score(label_true, label_pred)
    return (nmi + hom + com) / 3


# 聚类性能指标评估
def evaluate_clustering(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    vms = v_measure_score(y_true, y_pred)
    acc = getACC(y_true, y_pred)
    return ari, nmi, vms, acc


# 加载 DLPFC 数据集
def load_data(tissue_name, data_dir, truth_dir, img_dir):
    tissue_path = os.path.join(data_dir, tissue_name)
    adata_path = os.path.join(tissue_path, f"{tissue_name}_filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(adata_path)

    spatial_path = os.path.join(tissue_path, "spatial", "tissue_positions_list.csv")
    spatial_df = pd.read_csv(spatial_path, sep=",", header=None, index_col=0)
    adata.obs[["x1", "x2", "x3", "x4", "x5"]] = spatial_df[[1, 2, 3, 4, 5]]

    # 保留捕获点并标准化基因名
    adata = adata[adata.obs["x1"] == 1]
    adata.var_names = adata.var_names.str.upper()
    adata.var["genename"] = adata.var.index.astype("str")

    # 图像读取
    img_path = os.path.join(img_dir, tissue_name, "spatial", f"{tissue_name}_full_image.tif")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    img = cv2.imread(img_path)

    # 真实标签读取
    truth_path = os.path.join(truth_dir, f"{tissue_name}_truth.txt")
    truth_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
    truth_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = truth_df.loc[adata.obs_names, 'Ground Truth']

    # 聚类数设置
    cluster_map = {
        '151507': 7, '151508': 7, '151509': 7, '151510': 7,
        '151669': 5, '151670': 5, '151671': 5, '151672': 5,
        '151673': 7, '151674': 7, '151675': 7, '151676': 7
    }
    cluster_n = cluster_map.get(tissue_name, 5)

    coords = {
        'x_array': adata.obs["x2"].tolist(),
        'y_array': adata.obs["x3"].tolist(),
        'x_pixel': adata.obs["x4"].tolist(),
        'y_pixel': adata.obs["x5"].tolist()
    }

    return adata, img, coords, cluster_n


# 执行 SpaGCN 模型
def run_spagcn(adata, img, coords, cluster_n):
    x_array, y_array = coords['x_array'], coords['y_array']
    x_pixel, y_pixel = coords['x_pixel'], coords['y_pixel']

    pred, latent = spg.detect_spatial_domains_ez_mode(
        adata, img, x_array, y_array, x_pixel, y_pixel,
        n_clusters=cluster_n, histology=True, s=1, b=49, p=0.5,
        r_seed=100, t_seed=100, n_seed=100
    )
    adata.obs["pred"] = pd.Categorical(pred)
    adata.obsm["latent"] = latent

    # 域细化（可选）
    refined_pred = spg.spatial_domains_refinement_ez_mode(
        sample_id=adata.obs.index.tolist(),
        pred=pred,
        x_array=x_array,
        y_array=y_array,
        shape="hexagon"
    )
    adata.obs["refined_pred"] = pd.Categorical(refined_pred)

    return adata


# 保存结果
def save_result(adata, tissue_name, output_dir):
    output_path = os.path.join(output_dir, f"{tissue_name}_result.h5ad")
    os.makedirs(output_dir, exist_ok=True)
    adata.write(output_path)
    print(f"Saved result to: {output_path}")


# 单组织训练
def train_one_tissue(tissue_name, config):
    try:
        adata, img, coords, cluster_n = load_data(
            tissue_name,
            data_dir=config['data_dir'],
            truth_dir=config['truth_dir'],
            img_dir=config['img_dir']
        )

        adata = run_spagcn(adata, img, coords, cluster_n)

        # 评估
        obs_df = adata.obs.dropna()
        ari, nmi, vms, acc = evaluate_clustering(obs_df['Ground Truth'], obs_df['pred'])
        print(f"[{tissue_name}] ARI: {ari:.4f}, NMI: {nmi:.4f}, ACC: {acc:.4f}")

        # 保存
        raw_adata = sc.read_visium(
            os.path.join(config['data_dir'], tissue_name),
            count_file=f"{tissue_name}_filtered_feature_bc_matrix.h5"
        )
        raw_adata.obs[['pred', 'refined_pred', 'Ground Truth']] = adata.obs[['pred', 'refined_pred', 'Ground Truth']]
        raw_adata.obsm['latent'] = adata.obsm['latent']

        save_result(raw_adata, tissue_name, config['output_dir'])

        return ari, nmi, acc

    except Exception as e:
        print(f"Error processing {tissue_name}: {e}")
        return 0.0, 0.0, 0.0


# 主流程
def main():
    config = {
        'data_dir': "/home/jinjie/JinJie/DataAll/DLPFC",
        'truth_dir': "/home/jinjie/JinJie/DataAll/DLPFC/Truth_Label",
        'img_dir': "/home/jinjie/JinJie/DataAll/DLPFC",
        'output_dir': "/home/jinjie/JinJie/DataAll/DLPFC/SpaGCN"
    }

    section_ids = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676"
    ]

    ari_list, nmi_list, acc_list = [], [], []

    for tissue in section_ids:
        ari, nmi, acc = train_one_tissue(tissue, config)
        ari_list.append(ari)
        nmi_list.append(nmi)
        acc_list.append(acc)

    median_ari = np.median(ari_list)
    median_nmi = np.median(nmi_list)
    median_acc = np.median(acc_list)

    print(f"\nMedian Scores Across All Tissues:")
    print(f"ARI: {median_ari:.4f}, NMI: {median_nmi:.4f}, ACC: {median_acc:.4f}")


if __name__ == "__main__":
    set_random_seed(42)
    main()