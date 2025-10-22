import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class UTKFaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]


    def __len__(self):
        return len(self.image)


    def __getitem__(self, idx):
        image_name = self.image[idx]
        age, gender, *_ = image_name.split('_')
        image_path = os.path.join(self.folder_path, image_name)


        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        age = torch.tensor(float(age))
        gender = torch.tensor(int(gender))

        return image, age, gender

