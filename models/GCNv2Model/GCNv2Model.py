import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.nn as nn

from components.ResnetGCN.ResnetGCN import ResnetGCN


class GCNv2Model(nn.Module):
    """
    forward:
        - graph: Graph class.
    """

    def __init__(self, num_action_class=5):
        super(GCNv2Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = ResnetGCN()
        self.action_classify = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_action_class),
        )

    def forward(self, graph):
        feature = self.gcn(node_features=graph.nodes, edges=graph.edges)
        action_output = self.action_classify(feature)

        return action_output

    def evaluate(self):
        # Iterate through each module in the action_classify sequential module
        for module in self.action_classify:
            if isinstance(module, nn.Dropout):
                module.eval()
