import sys
import os

import numpy as np

project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.optim as optim
from tqdm import tqdm
from dataset.ActioinVideoDataset import ActionVideoDataset
from models.CNN3D.CNN3D import CNN3D
from torch import nn
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings("ignore")


# Prepare data
video_folder_path = os.path.join(project_path, "data/Short_Videos")
train_excel_path = os.path.join(project_path, "data/Short_Videos/annotation/train.xlsx")
validate_excel_path = os.path.join(
    project_path, "data/Short_Videos/annotation/validate.xlsx"
)
test_excel_path = os.path.join(project_path, "data/Short_Videos/annotation/test.xlsx")

train_dataset = ActionVideoDataset(video_folder_path, train_excel_path)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize your model
num_class = 5
model = CNN3D(num_class)

# Define loss function and optimizer
criterion_action = nn.CrossEntropyLoss()
criterion_danger = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch = int(
    open(os.path.join(project_path, "saved_models/CNN3D/current_epoch.txt"))
    .read()
    .strip()
)

if epoch > 0:
    previous_epoch = epoch - 1
    model.load_state_dict(
        torch.load(
            os.path.join(
                project_path,
                f"saved_models/CNN3D/epoch_{previous_epoch}/CNN3D_epoch_{previous_epoch}.pth",
            )
        )
    )
    optimizer.load_state_dict(
        torch.load(
            os.path.join(
                project_path,
                f"saved_models/CNN3D/epoch_{previous_epoch}/optimizer_epoch_{previous_epoch}.pth",
            )
        )
    )

model.to(model.device)  # Move model to device
print("##########Training with ", model.device)

# Training loop
while True:
    print(f"Training epoch {epoch}...")
    model.train()

    total_correct_action = 0
    total_correct_danger = 0
    total_samples = 0
    total_loss = 0

    for i, data in enumerate(tqdm(train_dataloader)):
        batch_video_index, batch_frames, batch_action_label, batch_danger_label = (
            data  # Assuming your data is in the format (inputs, labels)
        )
        #-------------------------------------------------------
        # Forward pass
        batch_frames = batch_frames.permute(0, 2, 1, 3, 4) # change torch[batch_size, 20, 224, 224, 3] to torch[batch_size, 3, 224, 224, 20]
        
        batch_frames = batch_frames.to(model.device)

        action_output, danger_output  = model(batch_frames)
        batch_action_label = batch_action_label.to(model.device)
        batch_danger_label = batch_danger_label.to(model.device)
        
        danger_output = danger_output.squeeze() # change torch[batch_size, 1] to torch[batch_size]

        loss_action = criterion_action(action_output, batch_action_label)
        loss_danger = criterion_danger(danger_output, batch_danger_label)
        #-------------------------------------------------------

        loss = loss_action + loss_danger

        # Result in train dataset
        # Compute accuracy for action
        _, predicted_action = torch.max(action_output, 1)
        total_correct_action += (predicted_action == batch_action_label).sum().item()

        # Compute accuracy for danger
        predicted_danger = (
            danger_output > 0.5
        ).float()  # assuming threshold 0.5
        total_correct_danger += (predicted_danger == batch_danger_label).sum().item()

        total_samples += batch_frames.size(0)

        total_loss += loss.item()

        optimizer.zero_grad()
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Calculate accuracy
    accuracy_action = total_correct_action / total_samples
    accuracy_danger = total_correct_danger / total_samples
    print(
        f"Epoch {epoch}: Loss {total_loss} Accuracy Action: {accuracy_action}, Accuracy Danger: {accuracy_danger}"
    )

    if epoch % 10 == 0:
        save_dir = os.path.join(
            project_path, f"saved_models/CNN3D/epoch_{epoch}/"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            model.state_dict(),
            os.path.join(
                project_path,
                f"saved_models/CNN3D/epoch_{epoch}/CNN3D_epoch_{epoch}.pth",
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                project_path,
                f"saved_models/CNN3D/epoch_{epoch}/optimizer_epoch_{epoch}.pth",
            ),
        )

        with open(
            os.path.join(
                project_path, f"saved_models/CNN3D/epoch_{epoch}/loss.txt"
            ),
            "w",
        ) as file:
            file.write(str(total_loss))

        with open(
            os.path.join(
                project_path, f"saved_models/CNN3D/epoch_{epoch}/accuracy.txt"
            ),
            "w",
        ) as file:
            file.write(str(accuracy_action) + "\n" + str(accuracy_danger))

        with open(
            os.path.join(project_path, "saved_models/CNN3D/current_epoch.txt"),
            "w",
        ) as file:
            file.write(str(epoch + 1))

    epoch += 1

print("Finished Training")