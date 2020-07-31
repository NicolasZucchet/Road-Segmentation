"""
Meta training on GoogleMaps data set using Reptile algorithm
Inspired from https://github.com/openai/supervised-reptile
In the entire script, parameters come from two different sources:
    - args: parameters that can be entered through command line
    - config: parameters that are defined in config.py. Some of them can be modified there.
"""

import argparse

# Project imports
from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationTask, RoadSegmentationDataset
from src.model import Model
from src.metrics import Hublot, report, intersection_over_union, f1_score, accuracy, create_saving_tools
from src.loss import BCEDicePenalizeBorderLoss
import config
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


ROOT_DIR = './data/GoogleMaps'


def load_model_transforms_indices(args):
    # Creates the model and load weights when given
    model = Model('UNet', config.IN_CHANNELS, config.N_CLASSES, device=config.device)
    model.load_weights(path=args.SAVE)

    # Get transformations from config
    transformations = { 'train': config.TRANSFORM_TRAIN, 'val': config.TRANSFORM_TEST }

    # Data will only be loaded when needed, just define which indices to take as train/val
    np.random.seed(config.RANDOM_SEED)
    random_permutation = np.random.permutation(config.N_SAMPLES_PER_TASK)
    indices_train, indices_val = random_permutation[:args.N_TRAIN], random_permutation[-args.N_VAL:]
    indices = { 'train': indices_train, 'val': indices_val }

    return model, indices, transformations


def optimizer_step(model, criterion, optimizer, data, args):
    """
    Inner loop of Reptile
    """
    model.train()
    for i in range(args.INNER_EPOCHS):
        train_loss, train_acc, ctr = 0., 0., 0.
        # For all the data
        for d in data:
            optimizer.zero_grad()
            outputs = model(d['images'])
            loss = criterion(outputs, d['labels'])
            loss.backward()
            optimizer.step()
            train_acc += ((torch.sigmoid(outputs) > 0.5) == d['labels']).float().mean()*outputs.shape[0]
            train_loss += loss.item()
            ctr += outputs.shape[0]
        train_loss /= ctr
        train_acc /= ctr
        print(f'\tEpoch: {i} Loss: {train_loss} Accuracy: {train_acc}')


def evaluate(model, criterion, data, hublot, args):
    """ 
    Evaluates the current model on the data
    """
    model.eval()
    with torch.no_grad():
        for d in data:
            outputs = model(d['images'])
            loss = criterion(outputs, d['labels']).item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            hublot.add_batch_results(preds, d['labels'], loss)


def train(model, indices, transformations, hublot, output_directory, args):
    criterion = BCEDicePenalizeBorderLoss()
    criterion.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.INNER_LR)

    best_mean_f1 = 0
        
    # Each meta iteration: training on one task randomly sampled
    for iteration in range(args.META_ITERATIONS):
        # Copy weights before meta update 
        weights = deepcopy(model.state_dict())

        lr_outer = args.OUTER_LR * (1 - iteration / args.META_ITERATIONS) # linear meta optimization schedule
        weights_updated = deepcopy(model.state_dict())
        # The past weights contribute for 1-lr_outer of the new weights
        weights_updated = { name : weights_updated[name]*(1-lr_outer) for name in weights_updated }

        hublot.set_phase('train')
        
        # Sample one of the training task
        task_names = random.sample(config.TRAIN_TASKS, args.TASK_BATCH_SIZE)
        for task_name in task_names:
            print(f'Meta iteration {iteration}, task: {task_name}')
            model.load_state_dict(weights)

            # Open the needed images
            task = RoadSegmentationTask(os.path.join(ROOT_DIR, task_name), indices['train'], indices['val'],
                device=config.device, train_transform=transformations['train'], val_transform=transformations['val'])
            train_data = DataLoader(task.train_data,  batch_size=args.BATCH_SIZE)
            val_data = DataLoader(task.val_data,  batch_size=args.BATCH_SIZE)

            # Perform inner_epochs gradient steps on training data
            optimizer_step(model, criterion, optimizer, train_data, args)

            # Evaluate on validation data of the task
            evaluate(model, criterion, val_data, hublot, args)

            # Interpolate between current weights and trained weights from this task
            weights_after = model.state_dict()
            # This task contributes for lr_outer/TASK_BATCH_SIZE of the weight update of this iteration
            weights_updated = { name : weights_updated[name] + lr_outer * weights_after[name]/args.TASK_BATCH_SIZE 
                for name in weights_updated }
        
        # Update model weights
        model.load_state_dict(weights_updated)
        del weights  # to avoid keeping the weights (potentially huge) in memory
        del weights_updated
            
        # Validation on all the validation tasks once in a while
        if iteration % int(10/args.TASK_BATCH_SIZE) == 0:
            hublot.set_phase('val')
            for task_name in config.VAL_TASKS:
                # For all validation task...
                task = RoadSegmentationTask(os.path.join(ROOT_DIR, task_name), indices['train'], indices['val'],
                    device=config.device, train_transform=transformations['train'], val_transform=transformations['val'])
                train_data =  DataLoader(task.train_data,  batch_size=args.BATCH_SIZE)
                val_data =  DataLoader(task.val_data,  batch_size=args.BATCH_SIZE)
                val_model = deepcopy(model)
                val_optimizer = torch.optim.Adam(val_model.parameters(), lr=args.INNER_LR)
                # ... train on training data ...
                optimizer_step(val_model, criterion, val_optimizer, train_data, args)
                # ... and evaluate on testing data
                evaluate(val_model, criterion, val_data, hublot, args)
        
        # saves model
        torch.save(model.state_dict(), output_directory+'/model.pt')

        hublot.save_epoch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='BATCH_SIZE', default=10, type=int, 
        help='Batch size (default: 10)')
    parser.add_argument('--task-batch-size', dest='TASK_BATCH_SIZE', default=1, type=int, 
        help='Number of tasks in one meta iteration (default: 1)')
    parser.add_argument('--save', dest='SAVE', default=None, type=str, 
        help='Path to model.pt (default: None, no trained model used)')
    parser.add_argument('--inner-epochs', dest='INNER_EPOCHS', type=int, default=1, 
        help='Number of epochs for inner optimization (default: 1)')
    parser.add_argument('--inner-lr', dest='INNER_LR', type=float, default=0.001, 
        help='Learning rate of inner optimization')
    parser.add_argument('--meta-iterations', dest='META_ITERATIONS', type=int, default=100, 
        help='Number of meta iterations (default: 100)')
    parser.add_argument('--outer-lr', dest='OUTER_LR', type=float, default=0.1, 
        help='Learning rate of outer optimization')
    parser.add_argument('--name', dest='NAME', type=str, default='EXPERIMENT', 
        help='Name of the experiment (default: EXPERIMENT)')
    parser.add_argument('--no-train', dest='NO_TRAIN', action='store_true', 
        help='To skip training phase')
    parser.add_argument('--n-train', dest='N_TRAIN', type=int, default=70,
        help='Number of training samples per task')
    parser.add_argument('--n-val', dest='N_VAL', type=int, default=30,
        help='Number of validation samples per task')

    args = parser.parse_args()

    model, indices, transformations = load_model_transforms_indices(args)
    hublot, output_directory = create_saving_tools(args)
    if not args.NO_TRAIN:
        train(model, indices, transformations, hublot, output_directory, args)
        hublot.close()