import sys
import os


project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.nn as nn

from torchvision.models import vit_b_16, ViT_B_16_Weights


class VisionTransformerModel(nn.Module):
    def __init__(self, num_action_class=5):
        super(VisionTransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads = nn.Linear(768, 512)
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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # Add batch size dimension
        feature = self.vit(x.float())
        action_output = self.action_classify(feature)
        danger_output = self.danger_classify(feature)

        return action_output, danger_output
