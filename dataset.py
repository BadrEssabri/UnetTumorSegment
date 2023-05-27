# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:53:42 2023

@author: Badr_
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)

        #convert to numpy array
        with h5py.File(file_path, 'r') as hf:
            # contains: a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) slice
            image = hf["image"][()].reshape(4,240,240)

            # only take the necrotic and non-enhancing tumor core (NCR/NET — label 1)
            mask = hf["mask"][()][:,:,0]

        return image, mask



