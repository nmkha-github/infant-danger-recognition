import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.nn as nn

from components.ResnetGCN.ResnetGCN import ResnetGCN


class GCNDropoutModel(nn.Module):
    """
    forward:
        - graph: Graph class.
    """

    def __init__(self, num_action_class=5):
        super(GCNDropoutModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = ResnetGCN()
        self.action_classify = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_action_class),
            nn.Softmax(dim=0),
        )

    def forward(self, graph):
        feature = self.gcn(node_features=graph.nodes, edges=graph.edges)
        action_output = self.action_classify(feature)

        return action_output

    def evaluate(self):
        self.action_classify[2].eval()
