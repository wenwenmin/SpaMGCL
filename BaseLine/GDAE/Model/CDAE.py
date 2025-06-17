import os
import random

import numpy as np
import torch
import torch.nn.modules.loss
from torch.backends import cudnn
from MyGDAE.GDAE.Model import GDAES


class Cdae:
    def __init__(self, adata, graph_dict,graph_topology, num_clusters,  device, config, roundseed=2024):
        seed = config['seed'] + roundseed
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
        self.graph_topology = graph_topology
        self.mode = config['mode']
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters


    def _start_(self):
        if self.mode == 'clustering':
            # X 是adata.obsm['X_pca']
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        elif self.mode == 'imputation':
            # X 是adata.X
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            raise Exception

        self.input_dim = self.X.shape[1]
        self.adj_dim = self.graph_dict.shape[1]
        self.topology_dim = self.graph_topology.shape[1]
        self.tisset_dim = self.X.shape[1]

        self.model = GDAES(self.num_clusters, self.tisset_dim,self.input_dim,self.adj_dim,self.topology_dim, self.model_config, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.train_config['lr'],
            weight_decay=float(self.train_config['decay']),  # 确保是float类型
        )
        self.model(self.X, self.graph_dict,self.graph_topology, self.optimizer)



    def trian(self):
        self._start_()


    def process(self):
        self.model.eval()
        recon = self.model.evaluate(self.X, self.graph_dict)
        return  recon