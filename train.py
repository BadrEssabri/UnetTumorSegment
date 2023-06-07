
from dataset import H5Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import numpy as np
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
NUM_EPOCHS = 20
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    model = UNET(in_channels=4, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    folder_path_train = './data/train_brain'
    dataset_train = H5Dataset(folder_path_train)
    train_loader = DataLoader(dataset_train)
    folder_path_val = './data/val_brain'
    dataset_val = H5Dataset(folder_path_val)
    val_loader = DataLoader(dataset_val)

    if LOAD_MODEL:
        load_checkpoint(torch.load("unet_tumor_weight.pth.tar"), model)

    val_loss = check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    early_stopper = EarlyStopper(patience=3, min_delta=0)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,filename="unet_tumor_weight.pth.tar")

        # check accuracy
        val_loss = check_accuracy(val_loader, model,loss_fn, device=DEVICE)

        if early_stopper.early_stop(val_loss):
            break
        # # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )


if __name__ == "__main__":
    main()
