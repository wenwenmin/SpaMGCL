import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)

        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)
        return x


class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)

def build_model(args):
    model = DeepGraphInfomax(
            hidden_channels=args.hidden_dims,
            encoder=Encoder(in_channels=args.spots_input_dims, hidden_channels=args.hidden_dims),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption
        )
    return model


