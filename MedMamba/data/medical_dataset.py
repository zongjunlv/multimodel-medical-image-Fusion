import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import nibabel as nib
from torch.utils import data
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, RandRotate90, RandFlip, RandGaussianNoise)



class Medical_Dataset(Dataset):
    def __init__(self,  mode, csv_path):
        self.mode = mode
        self.data = pd.read_csv(csv_path)
        self.train_aug = Compose([
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.01)
        ])
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        img = np.load(row["npy_path"])
        img = torch.from_numpy(img.copy()).float()
        if self.mode == 'train':
            img = self.train_aug(img)
        label = torch.as_tensor(int(row["label"]), dtype=torch.long)

        return img, label
