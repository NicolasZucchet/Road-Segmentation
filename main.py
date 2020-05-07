#!/usr/bin/env python3

# Project imports
from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationDataset
from src.model import UNet
from src.metrics import Hublot
# General imports
import numpy as np
import time, json
import os
import argparse
# Torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


IN_CHANNELS = 3
N_CLASSES = 1

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev)


def load_model_data(args):
    model = UNet(IN_CHANNELS, N_CLASSES)
    if args.SAVE is not None:
        model.load_state_dict(torch.load(args.SAVE))
    model.to(device)

    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)
    transform_train = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        MirroredRandomRotation(45),
        transforms.RandomResizedCrop(args.INPUT_SIZE, scale=(0.3,1)),
            # take a patch of size scale*input_size, and resize it to INPUT_SIZE
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ]
    transform_test = [
        transforms.Resize(args.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ]

    dataset_train = RoadSegmentationDataset('./data/training', indices=slice(70), train=True, transform=transform_train, device=device)
    dataset_valid = RoadSegmentationDataset('./data/training', indices=slice(70, 100), train=True, transform=transform_test, device=device)
    dataset_test = RoadSegmentationDataset('./data/test', train=False, transform=transform_test, device=device)
    data = {
        "train": DataLoader(dataset_train, batch_size=args.BATCH_SIZE),
        "valid": DataLoader(dataset_valid, batch_size=args.BATCH_SIZE),
        "test": DataLoader(dataset_test, batch_size=args.BATCH_SIZE)
    }

    return model, data

def create_saving_tools(args):
    output_directory = "experiment_results/" + args.NAME + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")
    os.makedirs(output_directory)
    hublot = Hublot(output_directory)  # Class that saves the results for Tensorboard
    print("\nExperiment results will be stored in ./"+output_directory)
    return hublot

def train(model, data, hublot, args):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR)
    since = time.time()
    best_f1 = 0.0

    for epoch in range(args.EPOCHS):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='BATCH_SIZE', default=10, type=int, help='Batch size (default: 10)')
    parser.add_argument('--save', dest='SAVE', default=None, type=str, help='Path to model.pt (default: None, no trained model used)')
    parser.add_argument('--input-size', dest='INPUT_SIZE', default=256, type=int, help='Size of the images given to the model (default: 256)')
    parser.add_argument('--lr', dest='LR', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', dest='EPOCHS', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--name', dest='NAME', type=str, default='EXPERIMENT', help='Name of the experiemnt (default: EXPERIMENT)')

    args = parser.parse_args()

    model, data = load_model_data(args)
    hublot = create_saving_tools(args)
    train(model, data, hublot, args)
    hublot.close()