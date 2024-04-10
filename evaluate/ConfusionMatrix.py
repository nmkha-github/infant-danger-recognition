import numpy as np

from dataset.ActioinVideoDataset import ActionVideoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def ConfusionMatrix(num_class, model, video_folder_path, excel_path):

    dataset = ActionVideoDataset(video_folder_path, excel_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    action_confusion_matrix = np.zeros((num_class, num_class))
    danger_confusion_matrix = np.zeros((2, 2))

    for i, data in enumerate(tqdm(dataloader)):
        batch_frames, batch_action_label, batch_danger_label = (
            data  # Assuming your data is in the format (inputs, labels)
        )
        batch_frames = batch_frames.to(model.device)  # shape [batch, 3, 224, 224]
        batch_action_label = batch_action_label.to(
            model.device
        )  # shape [batch, num_class]
        batch_danger_label = batch_danger_label.to(model.device)  # shape [batch]

        # Forward frame
        batch_outputs_action = torch.empty(0, num_class, requires_grad=True).to(
            model.device
        )  # num class = 5
        batch_outputs_danger = torch.empty(0, requires_grad=True).to(model.device)
        
        for _, frames in enumerate(batch_frames):
            np_frames = frames.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            graph = Graph(frames=np_frames)
            
            outputs_action, outputs_danger = model(
                graph=graph, context_frame=frames[len(frames) // 2]
            )  # example output: tensor([0.1, 0.2, 0.3, 0.2, 0.2]) and tensor([0.443])
            outputs_action = outputs_action.unsqueeze(0)
            batch_outputs_action = torch.cat((batch_outputs_action, outputs_action), 0)
            batch_outputs_danger = torch.cat((batch_outputs_danger, outputs_danger), 0)

        # # Loss function
        # loss_action = criterion_action(
        #     batch_outputs_action, batch_action_label
        # )  # Assuming labels are action classes
        # loss_danger = criterion_danger(
        #     batch_outputs_danger, batch_danger_label
        # )  # Assuming you have danger labels
        for j, batch_output_action in enumerate(batch_outputs_action):

            output_action_label = 0
            for label in range(num_class):
                if (batch_output_action[output_action_label] < batch_output_action[label]):
                    output_action_label = label
            # print(output_action_label, batch_action_label[j])
            action_confusion_matrix[output_action_label, batch_action_label[j]] += 1

            danger_confusion_matrix[int(batch_outputs_danger[j] >= 0.5), int(batch_danger_label[j])] += 1
        
    return action_confusion_matrix, danger_confusion_matrix