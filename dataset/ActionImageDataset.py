import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ActionImageDataset(Dataset):
    def __init__(self, image_folder, annotation_csv_file):
        self.image_folder = image_folder
        self.annotations = pd.read_csv(annotation_csv_file).filter(
            items=["index", "class", "danger"]
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_info = self.annotations.iloc[idx]
        image_index = image_info["index"]
        action_label = image_info["class"]
        danger_label = image_info["danger"]

        image_path = os.path.join(self.image_folder, f"img_{image_index}.jpg")
        image = Image.open(image_path)
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.PILToTensor()]
        )
        image = transform(image)

        image_index = torch.tensor(image_index, dtype=torch.long)
        action_label = torch.tensor(action_label, dtype=torch.long)
        danger_label = torch.tensor(danger_label, dtype=torch.float)

        return image_index, image, action_label, danger_label
