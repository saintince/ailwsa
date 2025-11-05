import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
        # height = torch.tensor(float(height))
        # weight = torch.tensor(float(weight))

        return image, age, gender

class BodyMDataset(Dataset):
    def __init__(self, images_folder, metadata_path, transform=None):
        self.images_folder = images_folder
        self.transform = transform

        self.metadata = pd.read_csv(metadata_path, sep='\t')


        self.metadata.dropna(subset=["height_cm", "weight_kg"], inplace=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_name = f"{row['subject_id']}.png"
        image_path = os.path.join(self.images_folder, image_name)

        # Загружаем изображение
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gender = 1 if row["gender"] == "male" else 0
        height = torch.tensor(float(row["height_cm"]))
        weight = torch.tensor(float(row["weight_kg"]))
        gender = torch.tensor(float(gender))

        return image, gender, height, weight


