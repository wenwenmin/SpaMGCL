import os
import numpy as np
import torch
import torch.nn.modules.loss
from tqdm import tqdm
from torch.backends import cudnn

from .Models import *  # Import model definitions from local module


class Mgcl:
    """
    Attributes:
        adata: Preprocessed AnnData object containing gene expression and spatial data.
        graph_dict: Dictionary containing graph information such as adjacency matrices.
        num_clusters: Number of clusters to be learned.
        device: Computation device ('cuda' or 'cpu').
        config: Configuration dictionary containing hyperparameters.
    """

    def __init__(self, adata, graph_dict, num_clusters, device, config, roundseed=0):
        self.mode = None

        # Set random seed for reproducibility
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
        """
        Initialize model inputs and load them onto the specified device.
        Prepare feature dimensions and instantiate the model.
        """
        # Convert PCA-reduced features to PyTorch tensor and move to device
        self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)

        # Move normalized and label adjacency matrices to device
        self.adj_norm = self.graph_dict["adj_norm"].to(self.device)
        self.adj_label = self.graph_dict["adj_label"].to(self.device)
        self.norm_value = self.graph_dict["norm_value"]

        # Define feature dimensions for encoder layers
        input_dim = self.X.shape[-1]  # Input feature dimension
        features_dims = [input_dim, self.config['hidden_dim'], self.config['latent_dim'], input_dim]

        self.input_dim = input_dim

        # Instantiate the Jin model using the provided configuration
        self.model = Jin(features_dims, self.config).to(self.device)

        # Setup Adam optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=float(self.config['weight_decay'])
        )

    def _fit_(self):
        """
        Training loop for the model over a fixed number of epochs.
        """
        pbar = tqdm(range(self.config['max_epoch']))  # Progress bar for training epochs
        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass: compute feature reconstruction loss and graph structure loss
            feat_loss, dig_loss = self.model(self.X, self.adj_norm)

            # Combine losses using lambda coefficient
            loss = (1 - self.config['lam']) * feat_loss + self.config['lam'] * dig_loss

            # Backward pass and optimization step
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            if self.config['gradient_clipping'] > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])

            self.optimizer.step()

            # Update progress bar with current loss values
            pbar.set_description(
                "Epoch {0}/{1} | Total Loss={2:.4f} | Feature Loss={3:.4f} | Dig Loss={4:.4f}".format(
                    epoch, self.config['max_epoch'], loss, feat_loss, dig_loss),
                refresh=True
            )

    def trian(self):  # Typo in function name; likely intended to be `train`
        """
        Start and train the model.
        """
        self._start_()
        self._fit_()

    def process(self):
        """
        Evaluate the model and return the final encoded representations.

        Returns:
            enc_rep: Encoded latent representations of the input data.
        """
        self.model.eval()
        enc_rep = self.model.evaluate(self.X, self.adj_norm)
        return enc_rep