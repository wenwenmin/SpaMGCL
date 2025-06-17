import os

import numpy as np
import torch
import torch.nn.modules.loss
from tqdm import tqdm
from torch.backends import cudnn

from MASK.Models import Jin


class MaskMAE:
    def __init__(self, adata, graph_dict, num_clusters,  device, config, roundseed=0):
        self.mode = None
        seed = config['seed'] + roundseed
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.graph_dict = graph_dict
        self.config = config
        self.num_clusters = num_clusters




    def _start_(self):

        self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.adj_norm = self.graph_dict["adj_norm"].to(self.device)
        self.adj_label = self.graph_dict["adj_label"].to(self.device)
        self.norm_value = self.graph_dict["norm_value"]
        input_dim = self.X.shape[-1]  # 输入特征维度
        num = self.X.shape[0]
        features_dims = [input_dim, self.config['hidden_dim'], self.config['latent_dim'], input_dim,num ]

        self.input_dim = self.X.shape[-1]
        self.model = Jin(features_dims, self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=float(self.config['weight_decay'])
        )

    def _fit_(self):
        pbar = tqdm(range(self.config['max_epoch']))
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            Lkss,ssss = self.model(self.X, self.adj_norm)
            loss = Lkss + ssss
            loss.backward()
            if self.config['gradient_clipping'] > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            self.optimizer.step()
            pbar.set_description(
             "Epoch {0}/{1} | Total Loss={2:.4f}  Total Loss={2:.4f} ".format(
                epoch, self.config['max_epoch'], Lkss,ssss),
            refresh=True)



    def trian(self):
        self._start_()
        self._fit_()

    def process(self):
        self.model.eval()
        enc_rep = self.model.evaluate(self.X, self.adj_norm)
        return enc_rep