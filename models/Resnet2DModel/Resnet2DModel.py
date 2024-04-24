import sys
import os


project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class Resnet2DModel(nn.Module):
    """
    forward:
        - graph: Graph class.
    """

    def __init__(self, num_action_class=5):
        super(Resnet2DModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False

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

        self.danger_classify = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        # Add batch size dimension
        feature = self.resnet(x.float())
        feature = torch.squeeze(feature)
        action_output = self.action_classify(feature)
        danger_output = self.danger_classify(feature)

        return action_output, danger_output
