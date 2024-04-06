import sys
import os

import numpy as np

project_path = os.getcwd()
sys.path.insert(0, project_path)

from components.Graph.Graph import Graph
import torch
import torch.optim as optim
from tqdm import tqdm
from dataset.ActioinVideoDataset import ActionVideoDataset
from models.SimpleModel.SimpleModel import SimpleModel
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
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize your model
num_class = 5
model = SimpleModel(num_class)

# Define loss function and optimizer
criterion_action = nn.CrossEntropyLoss()
criterion_danger = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load
initial_loss_action, initial_loss_danger = map(
    float,
    open(os.path.join(project_path, "saved_models/SimpleModel/init_parameter_loss.txt"))
    .read()
    .split(),
)

epoch = int(
    open(os.path.join(project_path, "saved_models/SimpleModel/current_epoch.txt"))
    .read()
    .strip()
)
previous_epoch = epoch - 1
model.load_state_dict(
    torch.load(
        os.path.join(
            project_path,
            f"saved_models/SimpleModel/epoch_{previous_epoch}/SimpleModel_epoch_{previous_epoch}.pth",
        )
    )
)
optimizer.load_state_dict(
    torch.load(
        os.path.join(
            project_path,
            f"saved_models/SimpleModel/epoch_{previous_epoch}/optimizer_epoch_{previous_epoch}.pth",
        )
    )
)
for param_group in optimizer.param_groups:
    param_group["lr"] = 0.01

model.to(model.device)  # Move model to device
print("##########Training with ", model.device)

dict_graph = {}
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
        batch_video_index = batch_video_index.to(model.device)
        batch_frames = batch_frames.to(model.device)  # shape [batch, 3, 224, 224]
        batch_action_label = batch_action_label.to(
            model.device
        )  # shape [batch, num_class]
        batch_danger_label = batch_danger_label.to(model.device)  # shape [batch]

        optimizer.zero_grad()

        # Forward frame
        batch_outputs_action = torch.empty(0, num_class, requires_grad=True).to(
            model.device
        )  # num class = 5
        batch_outputs_danger = torch.empty(0, requires_grad=True).to(model.device)
        for batch_index, frames in enumerate(batch_frames):
            graph = None
            if batch_video_index[batch_index].item() in dict_graph:
                graph = dict_graph[batch_video_index[batch_index].item()]
            else:
                np_frames = frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                graph = Graph(frames=np_frames)
                dict_graph[batch_video_index[batch_index].item()] = graph

            outputs_action, outputs_danger = model(
                graph=graph, context_frame=frames[len(frames) // 2]
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

        alpha = torch.tensor(2)
        lambda1 = torch.pow(loss_action.item() / initial_loss_action, alpha)
        lambda2 = torch.pow(loss_danger.item() / initial_loss_danger, alpha)

        loss = lambda1 * loss_action + lambda2 * loss_danger

        # Result in train dataset
        # Compute accuracy for action
        _, predicted_action = torch.max(batch_outputs_action, 1)
        total_correct_action += (predicted_action == batch_action_label).sum().item()

        # Compute accuracy for danger
        predicted_danger = (
            batch_outputs_danger > 0.5
        ).float()  # assuming threshold 0.5
        total_correct_danger += (predicted_danger == batch_danger_label).sum().item()

        total_samples += batch_frames.size(0)

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

    if epoch % 10 == 0:
        save_dir = os.path.join(
            project_path, f"saved_models/SimpleModel/epoch_{epoch}/"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            model.state_dict(),
            os.path.join(
                project_path,
                f"saved_models/SimpleModel/epoch_{epoch}/SimpleModel_epoch_{epoch}.pth",
            ),
        )
        torch.save(
            optimizer.state_dict(),
            os.path.join(
                project_path,
                f"saved_models/SimpleModel/epoch_{epoch}/optimizer_epoch_{epoch}.pth",
            ),
        )

        with open(
            os.path.join(
                project_path, f"saved_models/SimpleModel/epoch_{epoch}/loss.txt"
            ),
            "w",
        ) as file:
            file.write(str(total_loss))

        with open(
            os.path.join(
                project_path, f"saved_models/SimpleModel/epoch_{epoch}/accuracy.txt"
            ),
            "w",
        ) as file:
            file.write(str(accuracy_action) + "\n" + str(accuracy_danger))

        with open(
            os.path.join(project_path, "saved_models/SimpleModel/current_epoch.txt"),
            "w",
        ) as file:
            file.write(str(epoch + 1))

    epoch += 1

print("Finished Training")
