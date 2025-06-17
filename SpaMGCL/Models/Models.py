import copy
from functools import partial
import torch.nn.functional as F
import torch
from torch import nn, no_grad
from torch_geometric.nn import (
    DeepGraphInfomax,
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.nn.inits import reset, uniform


def create_activation(name):
    """
    Factory function to return activation module based on name.

    Args:
        name (str): Name of the activation function.

    Returns:
        nn.Module: Activation layer.
    """
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class GraphConv(nn.Module):
    """
    A generic graph convolutional layer supporting multiple types of GNN layers.
    """

    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn_layer = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn_layer(out_features)
        self.act = act
        self.dropout = dropout

        # Select GNN type
        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat":
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gin":
            self.conv = TransformerConv(in_channels=self.in_features, out_channels=self.out_features)
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


class Encoder(nn.Module):
    """
    Encoder network using GCN layers to learn latent node representations.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, bn=True, dropout_rate=0.1, act="prelu", bias=True):
        super(Encoder, self).__init__()
        bn_layer = nn.BatchNorm1d if bn else nn.Identity

        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim, bias=bias)
        self.bn1 = bn_layer(hidden_dim)

        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=latent_dim, bias=bias)
        self.bn2 = bn_layer(latent_dim)

        self.activation = create_activation(act)
        self.dropout = nn.Dropout(dropout_rate)

        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = self.activation(h)
        h = self.dropout(h)

        return h


class FeatureDecoder(nn.Module):
    """
    Decoder that reconstructs node features from latent representations.
    """

    def __init__(self, latent_dim, output_dim, dropout_rate=0.1, act="prelu", bias=True):
        super(FeatureDecoder, self).__init__()
        self.conv1 = GCNConv(in_channels=latent_dim, out_channels=output_dim, bias=bias)
        self.activation = create_activation(act)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        return h
class Jin(nn.Module):
    def __init__(self, features_dims, config, bn=False, att_dropout_rate=0.2, use_token=True):
        super(Jin, self).__init__()
        self.config = config
        [input_dim, hidden_dim, latent_dim, output_dim] = features_dims

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, bn=bn, dropout_rate=att_dropout_rate, act="prelu",
                               bias=True)

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, config['projection_dim']),
            nn.PReLU(),
            nn.Linear(config['projection_dim'], latent_dim),
        )

        self.weight = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        uniform(latent_dim, self.weight)

        self.use_token = use_token
        if self.use_token:
            self.pos_enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
            self.neg_enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))

        self.encoder_to_decoder = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)

        self.feat_decoder = FeatureDecoder(latent_dim, output_dim, dropout_rate=att_dropout_rate, act="prelu",
                                           bias=True)

        self.feat_loss = self.setup_loss_fn("sce", config['alpha'])

        self.feat_mask_rate = config['feat_mask_rate']

    def forward(self, feature, edge_index):
        x = feature
        use_pos_x, mask_nodes, keep_nodes = self.mask_feature(x, self.feat_mask_rate)
        use_neg_x = self.corrupt_feature(x, mask_nodes, keep_nodes)

        rep_pos_x = self.encoder(use_pos_x, edge_index)
        rep_neg_x = self.encoder(use_neg_x, edge_index)

        edfe = edge_index.clone()
        TE = rep_pos_x.clone()
        REP = self.projector(rep_pos_x[mask_nodes])
        RXP = self.projector(rep_neg_x[mask_nodes])

        s = self.avg_readout(TE, edfe, mask_nodes)
        dgi_loss = self.CL_Loss(REP, RXP, summary=s)

        rec_pos_x = self.encoder_to_decoder(rep_pos_x)
        torch.autograd.set_detect_anomaly(True)
        rec_pos_x = rec_pos_x.clone()
        rec_pos_x[mask_nodes] = 0
        rec_pos_x = self.feat_decoder(rec_pos_x, edge_index)

        feat_loss = self.feat_loss(x[mask_nodes], rec_pos_x[mask_nodes])

        return feat_loss, dgi_loss

    def reconstruct_adj_mse(self, g, emb):
        adj = g.to_dense()
        adj = adj.to(emb.device)
        res_adj = (emb @ emb.t())
        res_adj = F.sigmoid(res_adj)
        relative_distance = (adj * res_adj).sum() / (res_adj * (1 - adj)).sum()
        cri = torch.nn.MSELoss()
        res_loss = cri(adj, res_adj) + F.binary_cross_entropy_with_logits(adj, res_adj)
        loss = res_loss + relative_distance
        return loss

    def avg_readout(self, emb, adj=None, mask_nodes=None):
        adj = adj.to(dtype=emb.dtype)
        vsum = torch.mm(adj, emb)
        row_sum = torch.sum(adj, 1)
        if row_sum.is_sparse:
            row_sum = row_sum.to_dense()
        row_sum = row_sum.view(-1, 1)
        global_emb = vsum / row_sum
        global_emb = torch.sigmoid(global_emb)
        if mask_nodes is not None:
            global_emb = global_emb[mask_nodes]
        return global_emb

    def setup_loss_fn(self, loss_fn, alpha_l=2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def sce_loss(self, x, y, alpha):
        x_normalized = F.normalize(x, p=2, dim=-1)
        y_normalized = F.normalize(y, p=2, dim=-1)
        cosine_similarity = (x_normalized * y_normalized).sum(dim=-1)
        loss = (1 - cosine_similarity).pow(alpha)
        loss = loss.mean()
        return loss

    def discriminate(self, z, summary, sigmoid=True):
        assert isinstance(summary, torch.Tensor), "Summary should be a torch.Tensor"
        value = torch.matmul(z, torch.matmul(self.weight, summary.t()))
        return torch.sigmoid(value) if sigmoid else value

    def dgi_loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def CL_Loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        Cos_loss = -torch.log(1 - F.cosine_similarity(pos_z, neg_z) + 1e-15).mean()
        return Cos_loss + pos_loss + neg_loss

    def mask_feature(self, x, feat_mask_rate=0.3):
        if not (0 <= feat_mask_rate <= 1):
            raise ValueError(f"feat_mask_rate should be a value in [0, 1], but got {feat_mask_rate}.")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x should be a torch.Tensor, but got {type(x)}.")
        num_nodes = x.size(0)
        if num_nodes == 0:
            raise ValueError("The input feature matrix has 0 nodes, cannot perform masking.")
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(feat_mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        if self.use_token:
            if not hasattr(self, 'pos_enc_mask_token'):
                raise AttributeError("Attribute 'pos_enc_mask_token' must be defined when use_token is True.")
            out_x[mask_nodes] += self.pos_enc_mask_token
        else:
            out_x[mask_nodes] = 0.0
        return out_x, mask_nodes, keep_nodes

    def corrupt_feature(self, x, mask_nodes, keep_nodes):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x should be a torch.Tensor, but got {type(x)}.")
        if not isinstance(mask_nodes, torch.Tensor) or not isinstance(keep_nodes, torch.Tensor):
            raise TypeError("mask_nodes and keep_nodes should be torch.Tensor objects.")
        if x.device != mask_nodes.device or x.device != keep_nodes.device:
            raise ValueError("Input tensors x, mask_nodes, and keep_nodes must be on the same device.")
        CR = torch.zeros_like(x)
        if keep_nodes.numel() > 0:
            shuffled_indices = torch.randperm(keep_nodes.size(0), device=x.device)
            CR[keep_nodes] = x[keep_nodes][shuffled_indices]
        if self.use_token:
            if not hasattr(self, 'neg_enc_mask_token'):
                raise AttributeError("Attribute 'neg_enc_mask_token' must be defined when use_token is True.")
            CR[mask_nodes] += self.neg_enc_mask_token
        else:
            CR[mask_nodes] = 0.0
        return CR

    @torch.no_grad()
    def evaluate(self, datas, edges):
        enc_rep = self.encoder(datas, edges)
        return enc_rep
