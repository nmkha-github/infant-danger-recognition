import sys
import os

import numpy as np


project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import torch.optim as optim
from tqdm import tqdm
from dataset.ActionImageDataset import ActionImageDataset
from models.VisionTransformerModel.VisionTransformerModel import VisionTransformerModel
from torch import nn
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings("ignore")


# Prepare data
image_folder_path = os.path.join(project_path, "data/ASD_dataset")
train_csv_path = os.path.join(project_path, "data/ASD_dataset/annotation/train.csv")

train_dataset = ActionImageDataset(image_folder_path, train_csv_path)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize your model
num_class = 56
model = VisionTransformerModel(num_class)
model.to(model.device)  # Move model to device

# Define loss function and optimizer
criterion_action = nn.CrossEntropyLoss()
pos_weight = torch.tensor([50.0]).to(model.device)
criterion_danger = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

epoch = int(
    open(
        os.path.join(
            project_path, "saved_models/VisionTransformerModel/current_epoch.txt"
        )
    )
    .read()
    .strip()
)
if epoch > 0:
    previous_epoch = epoch - 1
    model.load_state_dict(
        torch.load(
            os.path.join(
                project_path,
                f"saved_models/VisionTransformerModel/epoch_{previous_epoch}/VisionTransformerModel_epoch_{previous_epoch}.pth",
            )
        )
    )
    optimizer.load_state_dict(
        torch.load(
            os.path.join(
                project_path,
                f"saved_models/VisionTransformerModel/epoch_{previous_epoch}/optimizer_epoch_{previous_epoch}.pth",
            )
        )
    )

print("##########Training with ", model.device)

dict_graph = {}
# Training loop
while True:
    print(f"Training epoch {epoch}...")
    model.train()

    total_correct_action = 0
    total_correct_danger = 0
    total_danger_samples = 0
    total_samples = 0
    total_loss = 0

    for i, data in enumerate(tqdm(train_dataloader)):
        batch_image_index, batch_image, batch_action_label, batch_danger_label = (
            data  # Assuming your data is in the format (inputs, labels)
        )
        batch_image_index = batch_image_index.to(model.device)
        batch_image = batch_image.to(model.device)  # shape [batch, 3, 224, 224]
        batch_action_label = batch_action_label.to(
            model.device
        )  # shape [batch, num_class]
        batch_danger_label = batch_danger_label.to(model.device).unsqueeze(
            1
        )  # shape [batch]

        optimizer.zero_grad()
        batch_outputs_action, batch_outputs_danger = model(batch_image)
        # Loss function
        loss_action = criterion_action(
            batch_outputs_action, batch_action_label
        )  # Assuming labels are action classes

        loss_danger = criterion_danger(
            batch_outputs_danger, batch_danger_label
        )  # Assuming you have danger labels

        loss = loss_action + loss_danger

        # Result in train dataset
        # Compute accuracy for action
        _, predicted_action = torch.max(batch_outputs_action, 1)
        total_correct_action += (predicted_action == batch_action_label).sum().item()

        # Compute accuracy for danger
        predicted_danger = (
            batch_outputs_danger > 0.5
        ).float()  # assuming threshold 0.5
        total_correct_danger += (predicted_danger == batch_danger_label).sum().item()

        total_samples += batch_image.size(0)

        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Calculate accuracy
    accuracy_action = total_correct_action / total_samples
    accuracy_danger = total_correct_danger / total_samples
    print(
        f"Epoch {epoch}: Loss {total_loss} Accuracy Action: {accuracy_action}, Accuracy Danger: {accuracy_danger}"
    )

    if epoch % 4 == 0:
        save_dir = os.path.join(
            project_path, f"saved_models/VisionTransformerModel/epoch_{epoch}/"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            model.state_dict(),
            os.path.join(
                project_path,
                f"saved_models/VisionTransformerModel/epoch_{epoch}/VisionTransformerModel_epoch_{epoch}.pth",
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                project_path,
                f"saved_models/VisionTransformerModel/epoch_{epoch}/optimizer_epoch_{epoch}.pth",
            ),
        )

        with open(
            os.path.join(
                project_path,
                f"saved_models/VisionTransformerModel/epoch_{epoch}/loss.txt",
            ),
            "w",
        ) as file:
            file.write(str(total_loss))

        with open(
            os.path.join(
                project_path,
                f"saved_models/VisionTransformerModel/epoch_{epoch}/accuracy.txt",
            ),
            "w",
        ) as file:
            file.write(str(accuracy_action) + "\n" + str(accuracy_danger))

        with open(
            os.path.join(
                project_path, "saved_models/VisionTransformerModel/current_epoch.txt"
            ),
            "w",
        ) as file:
            file.write(str(epoch + 1))

    epoch += 1

print("Finished Training")
