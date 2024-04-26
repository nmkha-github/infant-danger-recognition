import sys
import os


project_path = os.getcwd()
sys.path.insert(0, project_path)

import torch
from models.VisionTransformerModel.VisionTransformerModel import VisionTransformerModel
from dataset.ActionImageDataset import ActionImageDataset
from torch.utils.data import DataLoader


def compute_confusion_matrix(model, dataloader):
    # Initialize variables to accumulate counts
    tp_class0, fp_class0, tn_class0, fn_class0 = 0, 0, 0, 0
    tp_class1, fp_class1, tn_class1, fn_class1 = 0, 0, 0, 0

    # Set the model to evaluation mode
    model.eval()

    # Iterate over the dataset
    for (
        batch_image_index,
        batch_image,
        batch_action_label,
        batch_danger_label,
    ) in dataloader:
        batch_image = batch_image.to(model.device)
        batch_danger_label = batch_danger_label.to(model.device)

        # Forward pass
        with torch.no_grad():
            batch_outputs_action, batch_outputs_danger = model(batch_image)

        # Predictions
        predicted_danger = (batch_outputs_danger > 0.5).float()

        # Update confusion matrix counts for each class
        for pred, label in zip(predicted_danger, batch_danger_label):
            if label == 0:
                if pred == label:
                    tn_class0 += 1
                else:
                    fp_class0 += 1
            elif label == 1:
                if pred == label:
                    tp_class1 += 1
                else:
                    fn_class1 += 1

    # Calculate confusion matrix
    confusion_matrix = torch.tensor([[tn_class0, fp_class0], [fn_class1, tp_class1]])
    return confusion_matrix


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
epoch = 120
model_path = f"saved_models\VisionTransformerModel\epoch_{epoch}\VisionTransformerModel_epoch_{epoch}.pth"
num_class = 56
model = VisionTransformerModel(num_class)
model.eval()
model.to(model.device)
model.load_state_dict(torch.load(model_path))


def print_confusion_matrix(confusion_matrix):
    # Define class labels
    class_labels = ["Safe", "Danger"]

    # Print header
    print("\t\tPredicted Class")
    print("Actual Class\t" + "\t".join(class_labels))
    for i in range(len(class_labels)):
        print(
            f"{class_labels[i]}\t|\t{confusion_matrix[i][0]}\t{confusion_matrix[i][1]}"
        )


print(f"----------------------------EPOCH {epoch}---------------------------------")
# Compute confusion matrix for train dataset
confusion_matrix_train = compute_confusion_matrix(model, train_dataloader)
print("Confusion Matrix (Train):")
print_confusion_matrix(confusion_matrix_train)

# Compute confusion matrix for validation dataset
confusion_matrix_val = compute_confusion_matrix(model, val_dataloader)
print("Confusion Matrix (Validation):")
print_confusion_matrix(confusion_matrix_val)

# Compute confusion matrix for test dataset
confusion_matrix_test = compute_confusion_matrix(model, test_dataloader)
print("Confusion Matrix (Test):")
print_confusion_matrix(confusion_matrix_test)
