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
    global_max_pool
)
from torch_geometric.nn.inits import reset, uniform
from torch_scatter import scatter_add


def create_activation(name):
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
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
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
    def __init__(self, input_dim, hidden_dim, latent_dim, bn=True, dropout_rate=0.1, act="prelu", bias=True):
        super(Encoder, self).__init__()
        bn_layer = nn.BatchNorm1d if bn else nn.Identity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
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
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, bn=bn, dropout_rate=att_dropout_rate, act="prelu", bias=True)
        self.weight = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        uniform(latent_dim, self.weight)
        self.use_token = use_token
        if self.use_token:
            self.pos_enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
            self.neg_enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.encoder_to_decoder = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.feat_decoder = FeatureDecoder(latent_dim, output_dim, dropout_rate=att_dropout_rate, act="prelu", bias=True)
        self.feat_loss = self.setup_loss_fn("sce", config['alpha'])
        self.feat_mask_rate = config['feat_mask_rate']

    def forward(self, feature, edge_index):
        x = feature
        use_pos_x, mask_nodes, keep_nodes = self.mask_feature(x, self.feat_mask_rate)
        rep_pos_x = self.encoder(use_pos_x, edge_index)
        rec_pos_x = self.encoder_to_decoder(rep_pos_x)
        rec_pos_x[mask_nodes] = 0
        rec_pos_x = self.feat_decoder(rec_pos_x, edge_index)
        feat_loss = self.feat_loss(x[mask_nodes], rec_pos_x[mask_nodes])
        return feat_loss

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

    def mask_feature(self, x, feat_mask_rate=0.3):
        if not (0 <= feat_mask_rate <= 1):
            raise ValueError(f"feat_mask_rate 应为 [0, 1] 之间的值，但接收到 {feat_mask_rate}。")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x 应为 torch.Tensor 类型，但接收到 {type(x)}。")
        num_nodes = x.size(0)
        if num_nodes == 0:
            raise ValueError("输入特征矩阵的节点数为 0，无法执行掩码操作。")
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(feat_mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        if self.use_token:
            if not hasattr(self, 'pos_enc_mask_token'):
                raise AttributeError("需要定义 'pos_enc_mask_token' 属性以表示 Mask Token。")
            out_x[mask_nodes] += self.pos_enc_mask_token
        else:
            out_x[mask_nodes] = 0.0
        return out_x, mask_nodes, keep_nodes

    @torch.no_grad()
    def evaluate(self, datas, edges):
        enc_rep = self.encoder(datas, edges)
        return enc_rep