import sys
import os

project_path = os.getcwd()
sys.path.insert(0, project_path)

import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.VideoHelper.VideoHelper import VideoHelper


class ActionVideoDataset(Dataset):
    def __init__(self, video_folder, annotation_excel_file, transform=None):
        self.video_folder = video_folder
        self.annotations = pd.read_excel(annotation_excel_file).filter(
            items=["video", "action", "danger"]
        )
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_info = self.annotations.iloc[idx]
        video_index = video_info["video"]
        action_label = video_info["action"]
        danger_label = video_info["danger"]

        video_path = os.path.join(
            self.video_folder, f"Video_{video_index}_{action_label}.mp4"
        )
        frames = VideoHelper.extract_frames(video_path, 20)
        frames = np.array(frames)
        action_label = torch.tensor(action_label, dtype=torch.long)
        danger_label = torch.tensor(danger_label, dtype=torch.float)

        frames_tensor = torch.empty(frames.shape[0], 224, 224, 3)
        if self.transform:
            for idx, frame in enumerate(frames):
                frame = self.transform(frame).permute(1, 2, 0)
                frames_tensor[idx] = frame
        else:
            frames_tensor = torch.tensor(frames)

        # print(frames_tensor.shape)
        # print(action_label)
        # print(danger_label)

        f = open(
            os.path.join(
                project_path, "saved_models/SimpleModel/ActionVideoDataset_Log.txt"
            ),
            "a",
        )
        f.write(
            f"video {video_index} {frames_tensor.shape} action {action_label} danger {danger_label}\n"
        )
        f.close()
        return frames_tensor, action_label, danger_label
