import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.nn as nn
from torchvision import transforms

from components.ResnetCNN.ResnetCNN import ResnetCNN
from components.ResnetGCN.ResnetGCN import ResnetGCN


class FewNodeFeatureModel(nn.Module):
    """
    forward:
        - graph: Graph class.
        - context_frame: image tensor shape [3, 224, 224]
    """

    def __init__(self, num_action_class=5):
        super(FewNodeFeatureModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = ResnetGCN(in_channels=2)
        self.cnn = ResnetCNN()
        self.MLP = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.action_classify = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_action_class),
            nn.Softmax(dim=0),
        )
        self.danger_recognition = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, graph, context_frame):
        assert context_frame.shape == (
            3,
            224,
            224,
        ), "Reshape frame to (3, 224, 224)"

        # GCN flow
        feature1 = self.gcn(node_features=graph.nodes[:, :2], edges=graph.edges)

        # CNN flow
        context_frame = FewNodeFeatureModel.nomarlize_frame(context_frame)
        feature2 = self.cnn(context_frame)

        combine_feature = torch.cat((feature1, feature2), dim=0)
        MLP_output = self.MLP(combine_feature)

        action_output = self.action_classify(MLP_output)
        danger_output = self.danger_recognition(MLP_output)
        return action_output, danger_output

    @staticmethod
    def nomarlize_frame(frame):
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = transform(frame).unsqueeze(0)
        return image
