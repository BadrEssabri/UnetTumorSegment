
from dataset import H5Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Example usage
folder_path = './data/brains'
dataset = H5Dataset(folder_path)
dataloader = DataLoader(dataset)

i = 0
# Iterate over the dataloader
for images, masks in dataloader:
    # Perform training/validation/inference with the batches of images and masks
    #data
