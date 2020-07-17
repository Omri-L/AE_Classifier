import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob


class DatasetGeneratorAugTensors(Dataset):

    def __init__(self, path_to_tensor_files):
        self.all_files = glob.glob(path_to_tensor_files + "\\*.pth")

    def __getitem__(self, index):
        file_path = self.all_files[index]
        input_img, target_label = torch.load(file_path)

        return input_img, target_label

    def __len__(self):
        return len(self.all_files)
