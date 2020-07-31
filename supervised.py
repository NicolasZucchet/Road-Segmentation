"""
This script is made to train a network in the classical supervised learning way.

In the entire script, parameters come from two different sources:
    - args: parameters that can be entered through command line
    - config: parameters that are defined in config.py. Some of them can be modified there.
"""

# Project imports
from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationDataset
from src.model import Model
from src.metrics import Hublot, report, create_saving_tools
from src.loss import BCEDicePenalizeBorderLoss
import config
# General imports
import numpy as np
import time
import os
import argparse
# Torch imports
import torch
from torch.utils.data import DataLoader


def load_model_data(args):
    # Creates the model and load weights when given
    model = Model('UNet', config.IN_CHANNELS, config.N_CLASSES, device=config.device)
    model.load_weights(path=args.SAVE)
    # Freezes the first double convs of the model
    model.freeze(args.FREEZE)

    # Loads transformations for training and validation samples from config
    transform_train, transform_test = config.TRANSFORM_TRAIN, config.TRANSFORM_TEST

    # Randomly choose which samples will be used for training and which will be for validation
    if args.DATASET == "DiverCity":
        args.N_VAL = 0  # set it to 0 as not needed for DiverCity dataset
    np.random.seed(config.RANDOM_SEED)
    random_permutation = np.random.permutation(config.N_SAMPLES_PER_TASK)
    assert args.N_TRAIN + args.N_VAL <= config.N_SAMPLES_PER_TASK
    indices_train, indices_val = random_permutation[:args.N_TRAIN], random_permutation[-args.N_VAL:]
    if args.DATASET == "CIL":
        dataset_train = RoadSegmentationDataset(
            './data/CIL/training', indices=indices_train, train=True, transform=transform_train, device=config.device)
        dataset_valid = RoadSegmentationDataset(
            './data/CIL/training', indices=indices_val, train=True, transform=transform_test, device=config.device)
    elif args.DATASET == "DiverCity":
        # !!! For DiverCity dataset, n_val is not used
        # validation samples are not taken from the same tasks than training samples
        dataset_train = RoadSegmentationDataset('./data/DiverCity', train=True, indices=indices_train, subtasks=config.TRAIN_TASKS, 
            transform=transform_train, device=config.device)
        dataset_valid = RoadSegmentationDataset('./data/DiverCity', train=True, indices=indices_train, subtasks=config.VAL_TASKS, 
            transform=transform_test, device=config.device)
    else:
        raise ValueError("Dataset should be CIL or DiverCity")

    datasets = {
        "train": dataset_train,
        "val": dataset_valid
    }

    # Creates DataLoaders from the dataset previously defined
    data = { key: DataLoader(datasets[key],  batch_size=args.BATCH_SIZE) for key in datasets.keys() }

    return model, data, datasets


def train(model, data, hublot, output_directory, args):
    criterion = BCEDicePenalizeBorderLoss()
    criterion.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)
    since = time.time()
    best_f1 = 0.0

    for epoch in range(args.EPOCHS):
        print('Epoch', epoch)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
                    preds = (outputs > 0.5).float()  # binarize segmentation
                    hublot.add_batch_results(preds, e['labels'], loss.item())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

        # Saves the model if it's the best encountered so far
        epoch_val_f1 = hublot.get_metric('f1_score', 'val')  # get validation F1 score
        scheduler.step(epoch_val_f1)
        print('F1', epoch_val_f1)
        if epoch_val_f1 > best_f1:
            best_f1 = epoch_val_f1
            torch.save(model.state_dict(), output_directory+"/model.pt")
            
        hublot.save_epoch()

    time_elapsed = time.time() - since
    print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='BATCH_SIZE', default=10, type=int, 
        help='Batch size (default: 10)')
    parser.add_argument('--save', dest='SAVE', default=None, type=str, 
        help='Path to model.pt (default: None, no trained model used)')
    parser.add_argument('--lr', dest='LR', type=float, default=0.001, 
        help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', dest='EPOCHS', type=int, default=100, 
        help='Number of training epochs (default: 100)')
    parser.add_argument('--name', dest='NAME', type=str, default='EXPERIMENT', 
        help='Name of the experiment (default: EXPERIMENT)')
    parser.add_argument('--no-train', dest="NO_TRAIN", action='store_true', 
        help='To skip training phase')
    parser.add_argument('--freeze', dest="FREEZE", type=int, default=0, 
        help='Number of layers freezed (first layers)')
    parser.add_argument('--dataset', dest="DATASET", default="CIL", 
        help="Dataset to use, CIL or DiverCity")
    parser.add_argument('--n-train', dest="N_TRAIN", type=int, default=70, 
        help='Number of training samples')
    parser.add_argument('--n-val', dest="N_VAL", type=int, default=30, 
        help='Number of validation samples (will be set to 0 when using DiverCity dataset)')

    args = parser.parse_args()

    # Loads model and datasets
    model, data, datasets = load_model_data(args)
    # Creates the saving tools
    hublot, output_directory = create_saving_tools(args)

    # If training is needed
    if not args.NO_TRAIN:
        train(model, data, hublot, output_directory, args)
        hublot.close()

    # Report the performance of the model on the validation set after training
    report(model, datasets['val'], output_directory, args)
