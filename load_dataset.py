import torch
from torch import nn
from torch.utils.data import Dataset
import os
from PIL import Image

class PairedImageDataset(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.transform = transform
        self.hr_filenames = sorted([
            entry.path 
            for entry in os.scandir(hr_folder) 
            if entry.is_file()
        ])
        self.lr_filenames = sorted([
            entry.path 
            for entry in os.scandir(lr_folder) 
            if entry.is_file()
        ])

    def __len__(self):
        return len(self.hr_filenames)
    
    def __getitem__(self, index):
        hr_path = self.hr_filenames[index]
        lr_path = self.lr_filenames[index]
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        return hr_image, lr_image
    
class PairedImageDatasetTest(Dataset):
    def __init__(self, hr_folder, lr_folder, transform=None):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.transform = transform
        self.hr_filenames = sorted([
            entry.path 
            for entry in os.scandir(hr_folder) 
            if entry.is_file()
        ])
        self.lr_filenames = sorted([
            entry.path 
            for entry in os.scandir(lr_folder) 
            if entry.is_file()
        ])

    def __len__(self):
        return len(self.hr_filenames)
    
    def __getitem__(self, index):
        hr_path = self.hr_filenames[index]
        lr_path = self.lr_filenames[index]
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        return hr_image, lr_image, hr_path