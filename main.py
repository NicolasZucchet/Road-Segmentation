#!/usr/bin/env python3

# Project imports
from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationDataset
from src.model import Model
from src.metrics import Hublot, report
# General imports
import numpy as np
import time
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
    model = Model(args.MODEL_NAME, IN_CHANNELS, N_CLASSES, device=device)
    model.load_weights(path=args.SAVE)

    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)
    transform_train = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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

    if args.DATASET == "CIL":
        dataset_train = RoadSegmentationDataset('./data/CIL/training', indices=slice(70), train=True, transform=transform_train, device=device)
        dataset_valid = RoadSegmentationDataset('./data/CIL/training', indices=slice(70, 100), train=True, transform=transform_test, device=device)
        dataset_test = RoadSegmentationDataset('./data/CIL/test', train=False, transform=transform_test, device=device)
    elif args.DATASET == "GoogleMaps":
        dataset_train = RoadSegmentationDataset('./data/GoogleMaps', train=True, indices=slice(70), subtasks=True, 
            transform=transform_train, device=device)
        dataset_valid = RoadSegmentationDataset('./data/GoogleMaps', train=True, indices=slice(70, 100), subtasks=True, 
            transform=transform_test, device=device)
    else:
        raise ValueError("Dataset should be CIL or GoogleMaps")
    datasets = {
        "train": dataset_train,
        "valid": dataset_valid
    }
    if args.DATASET == "CIL":
        datasets["test"] = dataset_test

    data = { key: DataLoader(datasets[key],  batch_size=args.BATCH_SIZE) for key in datasets.keys() }

    return model, data, datasets

def create_saving_tools(args):
    output_directory = "experiment_results/" + args.NAME + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")
    os.makedirs(output_directory)
    hublot = None
    if not args.NO_TRAIN:
        hublot = Hublot(output_directory)  # Class that saves the results for Tensorboard
    print("\nExperiment results will be stored in ./"+output_directory)
    return hublot, output_directory

def train(model, data, hublot, output_directory, args):
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
            for e in data[phase]:
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
    parser.add_argument('--model-name', dest='MODEL_NAME', type=str, default='SegNet', help='Name of the model (default: SegNet)')
    parser.add_argument('--no-train', dest="NO_TRAIN", action='store_true', help='To skip training phase')
    parser.add_argument('--dataset', dest="DATASET", default="CIL", help="Dataset to use, CIL or GoogleMaps")

    args = parser.parse_args()

    model, data, datasets = load_model_data(args)
    hublot, output_directory = create_saving_tools(args)
    if not args.NO_TRAIN:
        train(model, data, hublot, output_directory, args)
        hublot.close()
    report(model, datasets['valid'], output_directory)
