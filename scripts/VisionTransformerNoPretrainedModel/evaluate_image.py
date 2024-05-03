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
from dataset.ActionImageDataset import ActionImageDataset
from models.VisionTransformerModel.VisionTransformerModel import VisionTransformerModel
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

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize your model
num_class = 56
dict_graph = {}

# Define loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion_action = nn.CrossEntropyLoss()
pos_weight = torch.tensor([50.0]).to(device)
criterion_danger = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def evaluate(model, dataloader, criterion_action, criterion_danger):
    total_correct_action = 0
    total_correct_danger = 0
    total_samples = 0
    total_action_loss = 0
    total_danger_loss = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            batch_video_index, batch_image, batch_action_label, batch_danger_label = (
                data
            )
            batch_video_index = batch_video_index.to(model.device)
            batch_image = batch_image.to(model.device)
            batch_action_label = batch_action_label.to(model.device)
            batch_danger_label = batch_danger_label.to(model.device).unsqueeze(
                1
            )  # shape [batch]

            batch_outputs_action, batch_outputs_danger = model(batch_image)

            # Loss function
            loss_action = criterion_action(
                batch_outputs_action, batch_action_label
            )  # Assuming labels are action classes

            loss_danger = criterion_danger(
                batch_outputs_danger, batch_danger_label
            )  # Assuming you have danger labels

            total_action_loss += loss_action.item()
            total_danger_loss += loss_danger.item()

            # Compute accuracy for action
            _, predicted_action = torch.max(batch_outputs_action, 1)
            total_correct_action += (
                (predicted_action == batch_action_label).sum().item()
            )

            # Compute accuracy for danger
            predicted_danger = (
                torch.sigmoid(batch_outputs_danger) > 0.5
            ).float()  # assuming threshold 0.5
            total_correct_danger += (
                (predicted_danger == batch_danger_label).sum().item()
            )

            total_samples += batch_image.size(0)

    accuracy_action = total_correct_action / total_samples
    accuracy_danger = total_correct_danger / total_samples
    return (
        total_action_loss / len(dataloader),
        accuracy_action,
        total_danger_loss / len(dataloader),
        accuracy_danger,
    )


# Directory containing saved models
models_dir = os.path.join(
    project_path, "saved_models/VisionTransformerNoPretrainedModel/evaluate"
)

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
    model_path = os.path.join(
        models_dir, epoch_dir, f"VisionTransformerNoPretrainedModel_epoch_{epoch}.pth"
    )
    model = VisionTransformerModel(num_class)
    model.eval()
    model.to(model.device)
    model.load_state_dict(torch.load(model_path))

    # Evaluate on train set
    action_loss, action_accuracy, danger_loss, danger_accuracy = evaluate(
        model, train_dataloader, criterion_action, criterion_danger
    )
    train_losses.append(action_loss)
    train_accuracies.append(action_accuracy)

    print(f"\nEpoch {epoch}:")
    print("  - Train:")
    print(
        f"       + Action loss: {action_loss:.4f}, Action accuracy: {action_accuracy:.4f}"
    )
    print(
        f"       + Danger loss: {danger_loss:.4f}, Danger accuracy: {danger_accuracy:.4f}"
    )

    # Evaluate on validation set
    action_loss, action_accuracy, danger_loss, danger_accuracy = evaluate(
        model, val_dataloader, criterion_action, criterion_danger
    )
    val_losses.append(action_loss)
    val_accuracies.append(action_accuracy)
    print("  - Validation:")
    print(
        f"       + Action loss: {action_loss:.4f}, Action accuracy: {action_accuracy:.4f}"
    )
    print(
        f"       + Danger loss: {danger_loss:.4f}, Danger accuracy: {danger_accuracy:.4f}"
    )

    # Evaluate on test set
    action_loss, action_accuracy, danger_loss, danger_accuracy = evaluate(
        model, test_dataloader, criterion_action, criterion_danger
    )
    test_losses.append(action_loss)
    test_accuracies.append(action_accuracy)
    print("  - Test:")
    print(
        f"       + Action loss: {action_loss:.4f}, Action accuracy: {action_accuracy:.4f}"
    )
    print(
        f"       + Danger loss: {danger_loss:.4f}, Danger accuracy: {danger_accuracy:.4f}"
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
plt.plot(epochs, test_accuracies, label="Test")
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
