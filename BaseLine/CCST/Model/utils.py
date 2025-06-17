# utils.py

import random
import numpy as np
import torch
import argparse
from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score

def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--model_name", type=str, default="ccst")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", type=str, default="../dataset/DLPFC")
    parser.add_argument("--dataset_name", type=str, default="151507")

    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--max_epoch", type=int, default=5000, help="number of training epochs")

    parser.add_argument("--hidden_dims", type=int, default=1024)
    parser.add_argument('--lambda_I', type=float, default=0.3)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getACC(label_key, prediction_key):
    NMI = normalized_mutual_info_score(label_key, prediction_key)
    HOM = homogeneity_score(label_key, prediction_key)
    COM = completeness_score(label_key, prediction_key)
    return (NMI + HOM + COM) / 3

