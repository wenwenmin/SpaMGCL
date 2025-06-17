import os
import warnings

# 忽略警告信息（谨慎使用）
warnings.filterwarnings("ignore")

# 标准库导入
from pathlib import Path

# 第三方库导入
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score

# 模型导入
from Model import DeepST

# 配置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = '/home/jinjie/miniconda3/envs/SpaMGCL/lib/R'
os.environ['R_USER'] = '/home/jinjie/miniconda3/envs/SpaMGCL/lib/python3.10/site-packages/rpy2'


def load_ground_truth(truth_file: Path, adata: sc.AnnData) -> sc.AnnData:
    """
    加载并合并真实标签到 AnnData 对象中。

    Parameters:
        truth_file (Path): Ground truth 文件路径
        adata (sc.AnnData): Scanpy 的 AnnData 对象

    Returns:
        sc.AnnData: 添加了 'Ground Truth' 列的 AnnData 对象
    """
    if not truth_file.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_file}")

    annotations = pd.read_csv(truth_file, sep='\t', header=None, index_col=0)
    annotations.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = annotations.loc[adata.obs_names, 'Ground Truth']
    return adata


def evaluate_adata(adata: sc.AnnData, cluster_key: str = 'DeepST_refine_domain') -> tuple[
    float, float, float, float, float]:
    """
    评估聚类结果：计算 ARI, NMI, HOM, COM, ACC

    Parameters:
        adata (sc.AnnData): 包含预测和真实标签的 AnnData
        cluster_key (str): 聚类结果列名

    Returns:
        tuple: ARI, NMI, HOM, COM, ACC
    """
    obs_df = adata.obs.dropna(subset=[cluster_key, 'Ground Truth'])
    true_labels = obs_df['Ground Truth']
    pred_labels = obs_df[cluster_key]

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = v_measure_score(true_labels, pred_labels)
    hom = homogeneity_score(true_labels, pred_labels)
    com = completeness_score(true_labels, pred_labels)
    acc = (nmi + hom + com) / 3

    return ari, nmi, hom, com, acc


def plot_spatial_domains(
        adata: sc.AnnData,
        output_path: str,
        metric_name: str = "ARI",
        metric_value: float = 0.0,
        cluster_key: str = "DeepST_refine_domain"
):
    """
    绘制空间域图，并在图上标注性能指标。

    Parameters:
        adata (sc.AnnData): 包含空间坐标的 AnnData
        output_path (str): 图像保存路径
        metric_name (str): 要显示的指标名称
        metric_value (float): 指标数值
        cluster_key (str): 聚类结果列名
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.spatial(
        adata,
        color=cluster_key,
        frameon=False,
        spot_size=150,
        ax=ax,
        show=False
    )
    ax.text(
        0.95, 0.05,
        f"{metric_name}: {metric_value:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def train_one_tissue(tissue_name: str, save_path: str = 'res/DLPFC'):
    """
    对单个组织切片进行训练和评估。

    Parameters:
        tissue_name (str): 组织切片名称（如 "151670"）
        save_path (str): 输出文件保存目录

    Returns:
        tuple[float, float]: ARI 和 ACC 分数，若失败则返回 (None, None)
    """
    try:
        # 设置域数量
        n_domains = 5 if tissue_name in ["151669", "151670", "151671", "151672"] else 7

        # 初始化模型
        dst = DeepST.run(save_path=save_path, task="Identify_Domain", pre_epochs=1000, epochs=500, use_gpu=True)

        # 构建路径
        base_path = Path("/home/jinjie/JinJie/JinJie/DataAll/DLPFC")
        data_dir = base_path / tissue_name
        count_file = data_dir / f"{tissue_name}_filtered_feature_bc_matrix.h5"
        truth_file = data_dir / f"{tissue_name}_truth.txt"

        # 加载数据
        adata = dst._get_adata(
            platform='Visium',
            data_path=str(base_path),
            data_name=tissue_name,
            count_file=count_file.name
        )

        # 数据增强与图像裁剪
        adata = dst._get_image_crop(adata, data_name=tissue_name)
        adata = dst._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

        # 构建空间图
        graph_dict = dst._get_graph(adata.obsm["spatial"], distType="BallTree")

        # 数据预处理
        data = dst._data_process(adata, pca_n_comps=200)

        # 模型训练
        deepst_embed = dst._fit(data=data, graph_dict=graph_dict)
        adata.obsm["DeepST_embed"] = deepst_embed

        # 聚类分析
        adata = dst._get_cluster_data(adata, n_domains=n_domains, priori=True)

        # 加载真实标签
        adata = load_ground_truth(truth_file, adata)

        # 评估性能
        ari, nmi, hom, com, acc = evaluate_adata(adata)

        print(f"Tissue: {tissue_name}, ARI: {ari:.4f}, ACC: {acc:.4f}")

        # 保存可视化结果
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, f'{tissue_name}_DeepST_results_domains.pdf')
        plot_spatial_domains(adata, plot_path, metric_name="ARI", metric_value=ari)

        # 保存处理后的 AnnData
        adata_file = os.path.join(save_path, f"{tissue_name}_DeepST_results.h5ad")
        adata.write(adata_file)

        return ari, acc

    except Exception as e:
        print(f"Error during processing tissue {tissue_name}: {e}")
        return None, None


def train_dlpfc():
    """
    批量训练多个 DLPFC 组织切片，并输出平均性能指标。
    """
    tissue_list = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676"
    ]
    save_path = "res/DLPFC"
    ari_list = []
    acc_list = []

    for tissue_name in tissue_list:
        print(f"\nProcessing tissue: {tissue_name}")
        ari, acc = train_one_tissue(tissue_name, save_path)
        if ari is not None and acc is not None:
            ari_list.append(ari)
            acc_list.append(acc)

    # 输出平均结果
    if ari_list and acc_list:
        avg_ari = sum(ari_list) / len(ari_list)
        avg_acc = sum(acc_list) / len(acc_list)
        print(f"\nAverage ARI: {avg_ari:.4f}, Average ACC: {avg_acc:.4f}")
    else:
        print("No valid results to calculate averages.")


if __name__ == "__main__":
    train_dlpfc()