import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


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


class ResnetGCN(nn.Module):
    def __init__(
        self,
        in_channels=12,
        out_channels=512,
        hidden_channels=[64, 64, 128, 128, 256, 256, 512],
    ):
        super(ResnetGCN, self).__init__()
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

    def forward(self, node_features, edges):
        x_tensor = torch.tensor(node_features, dtype=torch.float)
        edges_tensor = torch.tensor(edges, dtype=torch.long)
        data = Data(x=x_tensor, edge_index=edges_tensor.t().contiguous())
        data.x = F.normalize(data.x, p=2, dim=0)

        out = F.relu(self.conv1(data.x, data.edge_index))
        for block in self.blocks:
            out = block(out, data.edge_index)
        out = self.conv2(out, data.edge_index)
        # Apply global mean pooling to aggregate node features
        out = global_mean_pool(out, torch.zeros(out.size(0), dtype=torch.long))
        return out.squeeze()


# start_time = time.time()

# model = ResNetGCN(12, 256, hidden_channels=[64, 64, 128, 128, 256, 256, 512])
# output = model(graph.nodes, graph.edges)
# elapsed_time = time.time() - start_time  # Calculate elapsed time
# print(f"Elapsed time: {elapsed_time} seconds")

# print(output.shape)
# print(output)
# print(summary(model, graph.nodes, graph.edges))
