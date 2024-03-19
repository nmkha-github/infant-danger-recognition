import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Adjusts the identity mapping dimensions
        self.downsample_conv = None
        self.downsample_bn = None
        if in_channels != out_channels:
            self.downsample_conv = GCNConv(in_channels, out_channels)
            self.downsample_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        identity = x
        x = self.bn1(self.conv1(x, edge_index))
        x = F.relu(x)
        x = self.bn2(self.conv2(x, edge_index))
        x = F.relu(x)

        if self.downsample_conv is not None:
            identity = self.downsample_bn(
                self.downsample_conv(identity, edge_index=edge_index)
            )

        x += identity
        return x


class ResNetGCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=[]):
        super(ResNetGCN, self).__init__()
        if not hidden_channels:
            hidden_channels = [out_channels]
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_channels[i - 1], hidden_channels[i])
                for i in range(1, len(hidden_channels))
            ]
        )
        self.conv2 = GCNConv(hidden_channels[len(hidden_channels) - 1], out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        for block in self.blocks:
            x = block(x, edge_index)
        x = self.conv2(x, edge_index)
        # Apply global mean pooling to aggregate node features
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        return x
