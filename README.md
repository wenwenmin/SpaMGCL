# SpaMGCL

## Overview

The rapid development of spatial transcriptomics (ST) has enabled researchers to simultaneously capture gene expression profiles and spatial localization, offering unprecedented insights into tissue structure and function. However, spatial domain identification remains a critical challenge in ST data analysis, as current methods often struggle to fully capture the complex interplay between gene expression and spatial topology.
To address this challenge, we propose SpaMGCL (Spatial Masked Graph Contrastive Learning), a novel self-supervised learning framework that integrates a masking mechanism with a contrastive learning strategy to jointly optimize gene expression and spatial information for more discriminative latent representation learning. Specifically, SpaMGCL first generates contrastively augmented views through data perturbation and extracts feature representations using a shared encoder. A masking mechanism is then introduced to simulate missing data, encouraging the model to learn robust feature representations. Furthermore, to enhance feature discriminability, we design a neighborhood-aware contrastive learning module. This module aggregates local spatial information as anchors and leverages two distinct types of spatial neighborhoods as positive and negative samples, thereby maximizing intra-class similarity and enhancing inter-class separability. We comprehensively evaluated SpaMGCL on seven publicly available spatial transcriptomics datasets. Experimental results demonstrate that SpaMGCL consistently outperforms existing methods in spatial domain identification tasks, highlighting its effectiveness and robustness. Source code and all public datasets used in this paper are available at Github (\url{https://github.com/wenwenmin/SpaMGCL}) and Zenodo (\url{https://zenodo.org/records/15675248}).

![image-20250617152458856](E:\Desktop\SpaMGCL\assets\image-20250617152458856.png)

## Installation

	- NVIDIA GPU (a single Nvidia GeForce RTX 4090)
	- `pip install -r environment.txt`

## Data

 - All the datasets used in this paper can be downloaded from url：https://zenodo.org/records/15675248.

## Running demo

​	We can start the training process by running the Run.py file, and the training results can be read and visualized in the DLPFC_Plot.ipynb file.

## BaseLine

 - SpaGCN: A GCN-based method that integrates gene expression, spatial location, and histology to identify spatial domains and detect spatially variable genes.
 - SEDR：Integrates an autoencoder with a variational graph autoencoder to jointly embed gene expression and spatial information, thereby facilitating spatial transcriptomics analysis.
 - MaskGAE：A self-supervised graph autoencoder that leverages masked structural information to improve graph representation learning.
 - STAGATE：A graph attention autoencoder that adaptively learns spatial relationships, effectively improving spatial domain identification and denoising.
 - CCST：Unsupervised GNN clustering method that integrates gene expression and spatial location information for cell subtype and functional heterogeneity detection.
 - GraphST： A contrastive learning-based GNN model that integrates gene expression and spatial information to enhance spatial clustering and multi-sample integration.
 - GraphMAE： Masked graph autoencoder, using feature masking and cosine loss to strengthen self-supervised graph representation learning.
 - GDAE：A graph dual-view autoencoder that decouples node and aggregation views to capture both topology and attribute information for improved graph learning.
 - DeepST：A deep learning framework that integrates multimodal spatial transcriptomics data to correct batch effects and enhance spatial domain identification.

## Acknowledgements  

​	The work was supported in part by the National Natural Science Foundation of China (62262069) and Young Talent Program of Yunnan Province (C619300A067).

## Contact details

​	If you have any questions, please contact 2810127527@qq.com.













