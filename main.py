#!/usr/bin/env python3

from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationDataset
from src.model import UNet
import numpy as np
import time, json
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from src.metrics import Hublot


# Experiment variables
UNET_INPUT_SIZE = 256
BATCH_SIZE = 10
LR = 0.001
EPOCHS = 750
IN_CHANNELS = 3
N_CLASSES = 1
NAME = "BASIC_TRANSFORMS"
SAVE = True
MODEL_SAVE = None #"experiment_results/TEST_200429_165515/model.pt"

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev)

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
transform_train = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    MirroredRandomRotation(15),
    transforms.RandomResizedCrop(UNET_INPUT_SIZE, scale=(0.3,1)),
        # take a patch of size scale*input_size, and resize it to UNET_INPUT_SIZE
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
]
transform_test = [
    transforms.Resize(UNET_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
]

model = UNet(IN_CHANNELS, N_CLASSES)
if MODEL_SAVE is not None:
    model.load_state_dict(torch.load(MODEL_SAVE))
model.to(device)

dataset_train = RoadSegmentationDataset('./data/training', indices=slice(70), train=True, transform=transform_train, device=device)
dataset_valid = RoadSegmentationDataset('./data/training', indices=slice(70, 100), train=True, transform=transform_test, device=device)
dataset_test = RoadSegmentationDataset('./data/test', train=False, transform=transform_test, device=device)
data = {
    "train": DataLoader(dataset_train, batch_size=BATCH_SIZE),
    "valid": DataLoader(dataset_valid, batch_size=BATCH_SIZE),
    "test": DataLoader(dataset_test, batch_size=BATCH_SIZE)
}

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

output_directory = "experiment_results/" + NAME + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
if not os.path.exists("experiment_results"):
    os.makedirs("experiment_results")
os.makedirs(output_directory)
hublot = Hublot(output_directory)
print("\nExperiment results will be stored in ./"+output_directory)

since = time.time()
best_f1 = 0.0

for epoch in range(EPOCHS):
    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        hublot.set_phase(phase)

        # Iterate over data and labels (minibatches), by default, for one epoch.
        for i, e in enumerate(data[phase]):
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):  # grads computed only in the training phase
                outputs = model(e['images'])  # forward pass
                loss = criterion(outputs, e['labels'])
                outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
                preds = (outputs > 0.5).float()
                hublot.add_batch_results(preds, e['labels'], loss.item())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

    # Saves the model if its the best encountered so far
    epoch_val_f1 = hublot.get_metric('f1_score', 'valid')  # get validation F1 score
    if epoch_val_f1 > best_f1:
        best_f1 = epoch_val_f1
        torch.save(model.state_dict(), output_directory+"/model.pt")
        
    hublot.save_epoch()

time_elapsed = time.time() - since
print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

hublot.close()
# report(model, dataset_valid)