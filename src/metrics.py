"""
Contains functions to monitor training and validation
"""


import numpy as np
from torch.utils.data import DataLoader
from src.images import save_all, overlays
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
import torch
import json
import time, os


"""
IoU, Accuracy, F1 metrics defined to handle numpy 2D arrays as input
"""

def intersection_over_union(predictions, labels):
    predictions = (predictions > 0.5).astype(np.bool)
    labels = labels.astype(np.bool)
    intersection = predictions*labels
    union = predictions+labels
    return np.sum(intersection)/float(np.sum(union))

def accuracy(predictions, labels):
    predictions = (predictions > 0.5).astype(np.bool)
    labels = labels.astype(np.bool)
    return np.sum(predictions==labels)/labels.size

def f1_score(predictions, labels):
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1).astype(np.int)
    predictions = (predictions > 0.5).astype(np.int)
    return sklearn.metrics.f1_score(labels, predictions)


class Hublot:
    """
    Class that stores all the results obtained during an epoch and then save some metrics for Tensorboard
    """

    def __init__(self, output_directory):
        """
        Args:
            output_directory (string): where the results will be saved
        """
        self.writer = SummaryWriter(output_directory)
        self.epoch = 0
        self.phase = None
        self.phases = ['train', 'val']
        self._reset()

    def _reset(self):
        """
        Creates and empties all the structures for saving results
        """
        metrics = ['intersection_over_union', 'accuracy', 'f1_score']
        self.metrics = { metric: { phase: [] for phase in self.phases } for metric in metrics}
        self.losses = { phase: [] for phase in self.phases }

    def add_batch_results(self, predictions, labels, loss):
        """
        Receives predictions, labels and loss value for a batch and stores everything
        """
        predictions, labels = predictions.cpu().data.numpy(), labels.cpu().data.numpy()
        for metric in self.metrics:
            for i in range(predictions.shape[0]):
                self.metrics[metric][self.phase].append(
                    globals()[metric](predictions[i], labels[i])  # call the function that has metric as name
                )
        self.losses[self.phase].append(loss)
        
    def set_phase(self, phase):
        """
        Set phase in {train, val}
        """
        assert phase in self.phases
        self.phase = phase

    def save_epoch(self):
        """
        Saves (for Tensorboard) all the batch results received during an epoch
        """
        for metric in self.metrics:
            for phase in self.metrics[metric]:
                print(metric, phase, self.metrics[metric][self.phase])
                self.writer.add_scalar("Mean " + metric + "/" + phase, np.mean(self.metrics[metric][phase]), self.epoch)
        for phase in self.losses:
            self.writer.add_scalar("Loss/" + phase, np.mean(self.losses[phase]), self.epoch)
        self.epoch += 1
        self._reset()

    def get_metric(self, metric, phase):
        """
        Get the value of a metric for a given phase for the last epoch.
        Should be called at the end of an epoch, before save_epoch.
        
        Args:
            metric (string): should be in {intersection_over_union, accuracy, f1_score}
            phase (string): should be in {train, val}
        """
        if metric not in self.metrics:
            raise ValueError('metric not in ' + str(self.metrics.keys()))
        if phase not in ['train', 'val']:
            raise ValueError('phase not in {train, valid}')
        return np.mean(self.metrics[metric][phase])

    def close(self):
        self.writer.close()
        

def report(model, dataset, output_directory, args):
    """
    Evaluates model on dataset, saves predictions and, if labels are available computes some metrics. 
    Everything is stored in output_directory.
    args should have a BATCH_SIZE element.
    """
    data = DataLoader(dataset, batch_size=args.BATCH_SIZE)
    model.eval()  # set model in eval mode
    imgs, preds = [], []
    if dataset.train:
        gts = []
    for i, e in enumerate(data):
        outputs = model(e['images'])  # forward pass
        outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
        # images/outputs have shape [batch_size, n_channels, height, width]
        # transform them into list of [height, width, n_channels
        preds += list(outputs.permute(0, 2, 3, 1).cpu().data.numpy())
        imgs += list(e['raw_images'].permute(0, 2, 3, 1).cpu().data.numpy())  # use raw images, which are images before normalization
        if dataset.train:
            gts += list(e['labels'].permute(0, 2, 3, 1).cpu().data.numpy())  # use raw images, which are images before normalization

    preds, imgs = np.array(preds), np.array(imgs)
    if dataset.train:
        gts = np.array(gts)

    results = overlays(imgs, preds, alpha=0.4, binarize=True)
    save_all(results, output_directory+"/images")

    if dataset.train:
        metrics = {}
        metrics ['IOU'] = intersection_over_union(preds, gts)
        metrics['Accuracy'] = accuracy(preds, gts)
        metrics['F1'] = f1_score(preds, gts)
        for key in metrics:
            print(key, metrics[key])
        json.dump(metrics, open(output_directory+"/metrics.json", 'w'))


def create_saving_tools(args):
    """
    args should have attributes NAME, NO_TRAIN
    """
    # Directory where everything will be stored
    output_directory = "experiment_results/" + args.NAME + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")
    os.makedirs(output_directory)

    # If there is training, creates an Hublot instance to monitor it
    hublot = None
    if not args.NO_TRAIN:
        hublot = Hublot(output_directory)  # Class that saves the results for Tensorboard
    print("\nExperiment results will be stored in ./"+output_directory)

    return hublot, output_directory