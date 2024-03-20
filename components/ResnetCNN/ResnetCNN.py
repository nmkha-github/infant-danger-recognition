import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResnetCNN(nn.Module):
    def __init__(self):
        super(ResnetCNN, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
            return features.view(-1)
