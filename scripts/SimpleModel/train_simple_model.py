import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.optim as optim
from tqdm import tqdm
from dataset.ActioinVideoDataset import ActionVideoDataset
from models.SimpleModel.SimpleModel import SimpleModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import warnings

warnings.filterwarnings("ignore")


# Prepare data
video_folder_path = os.path.join(project_path, "data/Short_Videos")
train_excel_path = os.path.join(project_path, "data/Short_Videos/annotation/train.xlsx")
validate_excel_path = os.path.join(
    project_path, "data/Short_Videos/annotation/validate.xlsx"
)
test_excel_path = os.path.join(project_path, "data/Short_Videos/annotation/test.xlsx")

transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
)
train_dataset = ActionVideoDataset(
    video_folder_path, train_excel_path, transform=transform
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize your model
model = SimpleModel(5)
model.to(model.device)  # Move model to device

# Define initial losses
initial_loss_action = float("inf")
initial_loss_danger = float("inf")

# Define loss function and optimizer
criterion_action = nn.CrossEntropyLoss()
criterion_danger = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    max_loss = [0, 0]

    for i, data in enumerate(tqdm(train_dataloader)):
        batch_frames, batch_action_label, batch_danger_label = (
            data  # Assuming your data is in the format (inputs, labels)
        )
        batch_frames = batch_frames.to(model.device)
        batch_action_label = batch_action_label.to(model.device)
        batch_danger_label = batch_danger_label.to(model.device)

        optimizer.zero_grad()

        # Forward frame
        batch_outputs_action = torch.empty(0, 5, requires_grad=True).to(
            model.device
        )  # num class = 5
        batch_outputs_danger = torch.empty(0, requires_grad=True).to(model.device)
        for _, frames in enumerate(batch_frames):
            for idx, frame in enumerate(frames):
                if idx < frames.shape[0] - 1:
                    model(frame)
            outputs_action, outputs_danger = model(
                frame
            )  # example output: tensor([0.1, 0.2, 0.3, 0.2, 0.2]) and tensor([0.443])
            outputs_action = outputs_action.unsqueeze(0)
            batch_outputs_action = torch.cat((batch_outputs_action, outputs_action), 0)
            batch_outputs_danger = torch.cat((batch_outputs_danger, outputs_danger), 0)

        # Loss function
        loss_action = criterion_action(
            batch_outputs_action, batch_action_label
        )  # Assuming labels are action classes
        loss_danger = criterion_danger(
            batch_outputs_danger, batch_danger_label
        )  # Assuming you have danger labels

        if epoch == 0:
            initial_loss_action = loss_action.item()
            initial_loss_danger = loss_danger.item()

        alpha = torch.tensor(2)
        lambda1 = torch.pow(loss_action.item() / initial_loss_action, alpha)
        lambda2 = torch.pow(loss_danger.item() / initial_loss_danger, alpha)

        loss = lambda1 * loss_action + lambda2 * loss_danger
        max_loss = [
            max(max_loss[0], loss_action.item()),
            max(max_loss[0], loss_danger.item()),
        ]

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    if epoch == 0:
        f = open(
            os.path.join(
                project_path, f"saved_models/SimpleModel/init_parameter_loss.txt"
            ),
            "w",
        )
        f.write(str(max_loss))
        f.close()
    if epoch % 50 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                project_path, f"saved_models/SimpleModel/SimpleModel_epoch_{epoch}.pth"
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                project_path, f"saved_models/SimpleModel/optimizer_epoch_{epoch}.pth"
            ),
        )

print("Finished Training")
