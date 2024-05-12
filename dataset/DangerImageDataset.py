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


class DangerImageDataset(Dataset):
    def __init__(self, image_folder, annotation_csv_file):
        self.image_folder = image_folder
        self.annotations = pd.read_csv(annotation_csv_file).filter(
            items=["filename", "class"]
        )

        self.annotations = self.upsample_class1(self.annotations)

    def upsample_class1(self, df):
        class0 = df[df["class"] == 0]
        class1 = df[df["class"] == 1]
        ratio = len(class0) // len(class1)
        class1_upsampled = class1.copy()
        for _ in range(int(ratio - 1)):
            class1_upsampled = pd.concat([class1_upsampled, class1])
        return pd.concat([class0, class1_upsampled])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_info = self.annotations.iloc[idx]
        image_index = int(image_info["filename"])
        danger_label = image_info["class"]

        image_path = os.path.join(self.image_folder, f"image_{image_index}.jpg")
        image = Image.open(image_path)
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.PILToTensor()]
        )
        image = transform(image)

        image_index = torch.tensor(image_index, dtype=torch.long)
        danger_label = torch.tensor(danger_label, dtype=torch.float)

        return image_index, image, danger_label
