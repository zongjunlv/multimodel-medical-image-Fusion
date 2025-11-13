import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import nibabel as nib


import numpy as np
import pandas as pd

from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, Resize, RandRotate90, CenterSpatialCrop


class Medical_Dataset(Dataset):
    def __init__(self,  mode, target_size = (128,128,128)):
        if mode == 'train':
            self.csv_path = '/data02/workspace/LZJ_SPACE/dataset/ABUS_Classification/330_512_512/330_512_512_train.csv'
            self.transforms = Compose([
                EnsureChannelFirst(channel_dim='no_channel'),
                ScaleIntensity(),
                # CenterSpatialCrop(roi_size=(256, 256, 256)),
                Resize(target_size),
                RandRotate90()
            ])
        else:
            self.csv_path = '/data02/workspace/LZJ_SPACE/dataset/ABUS_Classification/330_512_512/330_512_512_val.csv'
            self.transforms = Compose([
                EnsureChannelFirst(channel_dim='no_channel'),
                ScaleIntensity(),
                # CenterSpatialCrop(roi_size=(256, 256, 256)),
                Resize(target_size),
            ])
        self.data = pd.read_csv(self.csv_path)
        
        self.class_map = {0:'benign', 1:'healthy', 2:'malignant'}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base_dir = "/data02/workspace/LZJ_SPACE"  # 或者 Path(self.csv_path).parent.parent 等动态求
        image_path = os.path.join(base_dir, self.data['file_path'][idx])
        image = nib.load(image_path).get_fdata(dtype=np.float32)

        row = self.data.iloc[idx]
        label_id = int(row["label"])
        # label = self.class_map[label_id]
        label = torch.tensor(label_id, dtype=torch.long)


        image = self.transforms(image)

        return image, label


