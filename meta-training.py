"""
Meta training using Reptile algorithm
Inspired from https://github.com/openai/supervised-reptile
"""

import argparse

# Project imports
from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationTask
from src.model import Model
from src.metrics import Hublot, report
# General imports
import numpy as np
import time
import random
import os
import argparse
from copy import deepcopy
# Torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)

IN_CHANNELS = 3
N_CLASSES = 1

ROOT_DIR = './data/GoogleMaps'
# Arbitrary choice
TRAIN_TASKS = [
    'Rome', 'London', 'Vancouver', 'Mexico_Countryside', 'NewDelhi', 
    'Sweden_Countryside', 'CapeTown', 'Tokyo', 'Stockholm', 'Shanghai', 
    'BuenosAires', 'Dubai', 'Riyadh', 'Switzerland_Countryside', 'Canada_Countryside', 
    'Zurich', 'NewYork', 'Teheran', 'Marakkesh', 'Egypt_Countryside', 'France_Countryside', 
    'Auckland', 'Sidney', 'MexicoCity', 'Italy_Countryside', 'Morocco_Countryside'
]
VAL_TASKS = [
    'Seattle', 'Bankok', 'SouthAfrica_Countryside', 'RioDeJaneiro', 'Miami', 
    'GB_Countryside', 'SanFrancisco', 'Paris', 'Cairo', 'Chicago'
]


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

    train_indices = slice(args.N_TRAIN)
    val_indices = slice(args.N_TRAIN, args.N_TRAIN+args.N_VAL)
    train_data = { task:
        RoadSegmentationTask(os.path.join(ROOT_DIR, task), train_indices, val_indices,
            device=device, train_transform=transform_train, val_transform=transform_test)
        for task in TRAIN_TASKS
    }
    val_data = { task:
        RoadSegmentationTask(os.path.join(ROOT_DIR,task), train_indices, val_indices,
            device=device, train_transform=transform_train, val_transform=transform_test)
        for task in VAL_TASKS
    }
    data = { 'train': train_data, 'val': val_data }

    return model, data


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


def optimizer_step(model, criterion, optimizer, data, args):
    """
    Inner loop of Reptile
    """
    model.train()
    for i in range(args.INNER_EPOCHS):
        for d in data:
            optimizer.zero_grad()
            pred = model(d['images'])
            loss = criterion(pred, d['labels'])
            loss.backward()
            optimizer.step()


def evaluate(model, criterion, data, args):
    loss, val_acc, ctr = 0., 0., 0.
    model.eval()
    with torch.no_grad():
        for d in data:
            pred = model(d['images'])
            loss += criterion(pred, d['labels']).item()
            val_acc += ((torch.sigmoid(pred) > 0.5) == d['labels']).float().mean()
            ## add f1 score 
            ctr += d['labels'].shape[0] / args.BATCH_SIZE
    loss /= ctr
    val_acc /= ctr
    return loss, val_acc


def train(model, data, hublot, output_directory, args):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.INNER_LR)
        
    # Sample an epoch by shuffling all training tasks
    for iteration in range(args.META_ITERATIONS):
        # Copy weights before meta update 
        weights_before = deepcopy(model.state_dict())
        
        # Sample one of the training task
        task_name = random.sample(TRAIN_TASKS, 1)[0]
        print(f'Iteration {iteration}: sampled {task_name}')
        task = data['train'][task_name]
        train_data =  DataLoader(task.train_data,  batch_size=args.BATCH_SIZE)
        val_data =  DataLoader(task.val_data,  batch_size=args.BATCH_SIZE)

        # Take inner_epochs gradient steps
        #optimizer_step(model, criterion, optimizer, train_data, args)

        # Evaluate on the training task
        #evaluate(model, criterion, val_data, args)  # TODO add hublot

        # Interpolate between current weights and trained weights from this task
        weights_after = model.state_dict()
        lr_outer = args.OUTER_LR * (1 - iteration / args.META_ITERATIONS) # linear meta optimization schedule
        model.load_state_dict({name : 
            weights_before[name] + (weights_after[name] - weights_before[name]) * lr_outer 
            for name in weights_before})
        del weights_before  # to avoid keeping the weights (potentially huge) in memory
            
        # Validation on all the validation tasks every 5 epochs
        if iteration % 5 == 0:
            for task_name in VAL_TASKS:
                task = data['val'][task_name]
                train_data =  DataLoader(task.train_data,  batch_size=args.BATCH_SIZE)
                val_data =  DataLoader(task.val_data,  batch_size=args.BATCH_SIZE)
                val_model = deepcopy(model)
                optimizer_step(val_model, criterion, optimizer, train_data, args)
                evaluate(val_model, criterion, val_data, args)
        
    
    
    print("k shot learning on CIL Dataset")
    k = 10 # set number of samples the model must learn test task from 
    train_loader, val_loader = test_task
    optimizer_step(model, loss, optimizer, train_loader, k, batch_size)
    loss, accuracy = evaluate(val_model, loss, val_loader, batch_size)
    print(f'K-Shot learning loss: {loss} accuracy: {accuracy}')
    
    return {'loss': { 'train': train_metalosses, 'test': val_metalosses},
            'accuracy': { 'train': train_accuracies, 'test': val_accuracies}}
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='BATCH_SIZE', default=10, type=int, 
        help='Batch size (default: 10)')
    parser.add_argument('--save', dest='SAVE', default=None, type=str, 
        help='Path to model.pt (default: None, no trained model used)')
    parser.add_argument('--input-size', dest='INPUT_SIZE', default=256, type=int, 
        help='Size of the images given to the model (default: 256)')
    parser.add_argument('--inner-epochs', dest='INNER_EPOCHS', type=int, default=1, 
        help='Number of epochs for inner optimization (default: 1)')
    parser.add_argument('--inner-lr', dest='INNER_LR', type=float, default=0.0004, 
        help='Learning rate of inner optimization')
    parser.add_argument('--outer-lr', dest='OUTER_LR', type=float, default=0.1, 
        help='Learning rate of outer optimization')
    parser.add_argument('--meta-iterations', dest='META_ITERATIONS', type=int, default=1000, 
        help='Number of meta iterations (default: 1000)')
    parser.add_argument('--name', dest='NAME', type=str, default='EXPERIMENT', 
        help='Name of the experiemnt (default: EXPERIMENT)')
    parser.add_argument('--model-name', dest='MODEL_NAME', type=str, default='SegNet', 
        help='Name of the model (default: SegNet)')
    parser.add_argument('--no-train', dest="NO_TRAIN", action='store_true', 
        help='To skip training phase')
    parser.add_argument('--n-train', dest="N_TRAIN", type=int,
        help='Number of training samples per task')
    parser.add_argument('--n-val', dest="N_VAL", type=int,
        help='Number of validation samples per task')

    args = parser.parse_args()

    model, data = load_model_data(args)
    hublot, output_directory = create_saving_tools(args)
    if not args.NO_TRAIN:
        train(model, data, hublot, output_directory, args)
        hublot.close()
    report(model, datasets['valid'], output_directory)