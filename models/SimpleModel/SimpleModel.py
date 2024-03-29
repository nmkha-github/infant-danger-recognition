import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from components.Graph.Graph import Graph
from components.ResnetCNN.ResnetCNN import ResnetCNN
from components.ResnetGCN.ResnetGCN import ResnetGCN


class SimpleModel(nn.Module):
    def __init__(self, num_action_class=5):
        super(SimpleModel, self).__init__()
        self.graph = Graph()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = ResnetGCN()
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
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256, num_action_class),
            nn.Softmax(dim=0),
        )
        self.danger_recognition = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, tensor_frame):
        with torch.no_grad():
            assert tensor_frame.shape == (224, 224, 3), "Reshape frame to (224, 224, 3)"
            np_frame = np.array(tensor_frame.cpu(), dtype="uint8")

            # GCN flow
            self.graph.append(np_frame)
            feature1 = self.gcn(node_features=self.graph.nodes, edges=self.graph.edges)

            # CNN flow
            context_frame = SimpleModel.nomarlize_frame(np_frame).to(self.device)
            feature2 = self.cnn(context_frame)

            combine_feature = torch.cat((feature1, feature2), dim=0)
            MLP_output = self.MLP(combine_feature)

            action_output = self.action_classify(MLP_output)
            danger_output = self.danger_recognition(MLP_output)
            return action_output, danger_output

    @staticmethod
    def nomarlize_frame(image):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = transform(image).unsqueeze(0)
        return image


# model = SimpleModel(5).to(device)
# start_time = time.time()
# action, danger = model(torch.tensor(img).to(device))
# elapsed_time = time.time() - start_time  # Calculate elapsed time
# print(f"SimpleModel elapsed time: {elapsed_time} seconds")

# print(action)
# print(danger)
