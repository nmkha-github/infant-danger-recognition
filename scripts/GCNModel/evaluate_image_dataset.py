import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from components.Graph.Graph import Graph
from dataset.ActionImageDataset import ActionImageDataset
from models.GCNModel.GCNModel import GCNModel
from torch import nn


# Set project path
project_path = os.getcwd()

# Prepare data
image_folder_path = os.path.join(project_path, "data/ASD_dataset")
train_csv_path = os.path.join(project_path, "data/ASD_dataset/annotation/train.csv")
val_csv_path = os.path.join(project_path, "data/ASD_dataset/annotation/validate.csv")
test_csv_path = os.path.join(project_path, "data/ASD_dataset/annotation/test.csv")

train_dataset = ActionImageDataset(image_folder_path, train_csv_path)
val_dataset = ActionImageDataset(image_folder_path, val_csv_path)
test_dataset = ActionImageDataset(image_folder_path, test_csv_path)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize your model
num_class = 56
dict_graph = {}

# Define loss function
criterion_action = nn.CrossEntropyLoss()


def evaluate(model, dataloader, criterion_action):
    model.eval()
    total_correct_action = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            batch_video_index, batch_image, batch_action_label, batch_danger_label = (
                data
            )
            batch_video_index = batch_video_index.to(model.device)
            batch_image = batch_image.to(model.device)
            batch_action_label = batch_action_label.to(model.device)

            batch_outputs_action = torch.empty(0, num_class, requires_grad=False).to(
                model.device
            )
            for batch_index, image in enumerate(batch_image):
                graph = None
                if batch_video_index[batch_index].item() in dict_graph:
                    graph = dict_graph[batch_video_index[batch_index].item()]
                else:
                    np_frames = np.array(
                        [
                            image.permute(1, 2, 0)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)[:, :, ::-1]
                        ]
                    )
                    graph = Graph(frames=np_frames)
                    dict_graph[batch_video_index[batch_index].item()] = graph

                outputs_action = model(graph=graph)
                outputs_action = outputs_action.unsqueeze(0)
                batch_outputs_action = torch.cat(
                    (batch_outputs_action, outputs_action), 0
                )

            loss_action = criterion_action(batch_outputs_action, batch_action_label)
            total_loss += loss_action.item()

            _, predicted_action = torch.max(batch_outputs_action, 1)
            total_correct_action += (
                (predicted_action == batch_action_label).sum().item()
            )
            total_samples += batch_image.size(0)

    accuracy_action = total_correct_action / total_samples
    return total_loss / len(dataloader), accuracy_action


# Directory containing saved models
models_dir = os.path.join(project_path, "saved_models/ImageData/evaluate")

# Lists to store loss and accuracy values
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

epochs = []

# Evaluate models
for epoch_dir in tqdm(
    sorted(os.listdir(models_dir), key=lambda x: int(x.split("_")[1]))
):
    epoch = int(epoch_dir.split("_")[1])
    print(f"\nEvaluating epoch {epoch}...")
    model_path = os.path.join(models_dir, epoch_dir, f"ImageData_epoch_{epoch}.pth")
    model = GCNModel(num_class)
    model.to(model.device)
    model.load_state_dict(torch.load(model_path))

    # Evaluate on train set
    train_loss, train_accuracy = evaluate(model, train_dataloader, criterion_action)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Evaluate on validation set
    val_loss, val_accuracy = evaluate(model, val_dataloader, criterion_action)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Evaluate on test set
    test_loss, test_accuracy = evaluate(model, test_dataloader, criterion_action)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
    )
    print(
        f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    )
    print(
        f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

    epochs.append(epoch)

# Plot loss curve
plt.plot(epochs, train_losses, label="Train")
plt.plot(epochs, val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("train_val_loss.png")
plt.clf()

# Plot accuracy curve
plt.plot(epochs, train_accuracies, label="Train")
plt.plot(epochs, val_accuracies, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("train_val_accuracy.png")
plt.clf()

plt.plot(epochs, test_losses, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("test_loss.png")
plt.clf()

plt.plot(epochs, test_accuracies, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("test_accuracy.png")
plt.clf()
