import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 标准库导入
from pathlib import Path

# 第三方库导入
import scanpy as sc
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

# 自定义模块导入
from CCST.RUN_Train import train
from CCST.utils import build_args


def load_melanoma_data(data_path: Path):
    """加载并验证 Melanoma 数据集"""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    return adata


def load_ground_truth(truth_path: Path, adata: sc.AnnData):
    """加载并匹配真实标签"""
    if not truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_path}")

    annotations_df = pd.read_csv(truth_path, index_col=0)
    annotations_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = annotations_df.loc[adata.obs_names, 'Ground Truth']
    return adata


def evaluate_clustering(adata: sc.AnnData):
    """评估聚类结果，计算 ARI 和 NMI"""
    obs_df = adata.obs.dropna()
    true_labels = obs_df['Ground Truth']
    predicted_labels = obs_df['mclust']

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    return ari, nmi


def process_hmelanoma(args):
    """主流程函数：配置参数 -> 加载数据 -> 训练模型 -> 评估性能"""
    args.dataset_name = 'HM'
    n_clusters = 4  # 根据 HM 数据集特点设置簇的数量

    # 文件路径定义
    data_path = Path("/home/jinjie/JinJie/DongTing/DataAll/HM/reading_h5/Melanoma_filtered_feature_bc_matrix.h5")
    truth_path = Path("/home/jinjie/JinJie/DongTing/DataAll/HM/manual_annotations.csv")

    # 加载数据
    adata = load_melanoma_data(data_path)

    # 加载 Ground Truth
    adata = load_ground_truth(truth_path, adata)

    # 模型训练
    adata = train(args, adata, n_clusters)

    # 性能评估
    ari, nmi = evaluate_clustering(adata)

    return ari, nmi


def train_hmelanoma():
    """配置参数并运行训练流程"""
    args = build_args()
    args.top_genes = 2000  # 基因特征数量
    args.radius = 1  # 半径参数，根据实验调整

    ari, nmi = process_hmelanoma(args)

    print(f"Median ARI: {ari:.4f}, Median NMI: {nmi:.4f}")


if __name__ == '__main__':
    train_hmelanoma()