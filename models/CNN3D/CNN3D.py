import torch
import torch.nn as nn
from torchvision import transforms

class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super(CNN3D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_vector = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )

        self.action_classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28 * 2, 512),  # Calculate the input size after max pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=0)
        )

        self.danger_recognition = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28 * 2, 512),  # Calculate the input size after max pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, context_frame):
        # context_frame = CNN3D.nomarlize_frame(context_frame)
        
        feature_vector = self.feature_vector(context_frame)
        feature_vector = feature_vector.view(-1, 128 * 28 * 28 * 2)  # Flatten

        action_output = self.action_classify(feature_vector)
        danger_output = self.danger_recognition(feature_vector)
        return action_output, danger_output

    # def nomarlize_frame(frames):
    #     transform = transforms.Compose(
    #         [
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #             ),
    #         ]
    #     )

    #     frames = transform(frames).unsqueeze(0)
    #     return frames



# model = CNN3D(5)
# test_data = torch.rand(4, 3, 224, 224, 20)
# action_output, danger_output = model(test_data)
# print(action_output.shape, danger_output.shape)