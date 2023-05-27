# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:53:42 2023

@author: Badr_
"""
import matplotlib.pyplot as plt 
import h5py
filename = "volume_100_slice_60.h5"

with h5py.File(filename, "r") as f:
    image_brain = f["image"][()]  
    mask_brain = f["mask"][()]

plt.imshow(image_brain[:,:,0], cmap='gray')
cmap_mask = plt.cm.get_cmap('Reds')
mask_alpha = 0.4 # Set the alpha (transparency) channel for the mask
plt.imshow(mask_brain[:,:,0], cmap=cmap_mask, alpha=mask_alpha) # Overlay the mask on the brain image

plt.figure()
plt.imshow(image_brain[:,:,1], cmap='gray')
plt.imshow(mask_brain[:,:,1], cmap=cmap_mask, alpha=mask_alpha)

plt.figure()
plt.imshow(image_brain[:,:,2], cmap='gray')
plt.imshow(mask_brain[:,:,2], cmap=cmap_mask, alpha=mask_alpha)

plt.figure()
plt.imshow(image_brain[:,:,3], cmap='gray')
plt.imshow(mask_brain[:,:,2], cmap=cmap_mask, alpha=mask_alpha)

