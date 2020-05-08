import numpy as np
from torch.utils.data import DataLoader
from src.images import save_all, overlays
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
import torch
import json


def intersection_over_union(predictions, labels):
    predictions = (predictions > 0.5).astype(np.bool)
    labels = labels.astype(np.bool)
    intersection = predictions*labels
    union = predictions+labels
    return np.sum(intersection)/float(np.sum(union))

def accuracy(predictions, labels):
    predictions = (predictions > 0.5).astype(np.bool)
    labels = labels.astype(np.bool)
    return np.sum(predictions*labels)/float(labels.shape[0])

def f1_score(predictions, labels):
    return sklearn.metrics.f1_score(labels, predictions)


class Hublot:
    """
    Class that can store all the results obtained during an epoch and then save everything for Tensorboard
    """

    def __init__(self, output_directory):
        """
        Args:
            output_directory (string): where the results will be saved
            n_classes (int): number of classes
        """
        self.writer = SummaryWriter(output_directory)
        self._reset()
        self.epoch = 0
        self.phase = None

    def _reset(self):
        metrics = ['intersection_over_union', 'accuracy', 'f1_score']
        self.metrics = { metric:{'train': [], 'valid': []} for metric in metrics}
        self.losses = { 'train': [], 'valid': [] }

    def add_batch_results(self, predictions, labels, loss):
        predictions, labels = predictions.cpu().data.numpy(), labels.cpu().data.numpy()
        # reshape to get [n_images, n_pixels] array
        predictions = predictions.reshape(predictions.shape[0], -1)
        labels = labels.reshape(predictions.shape[0], -1)
        for metric in self.metrics:
            for i in range(predictions.shape[0]):
                self.metrics[metric][self.phase].append(
                    globals()[metric](predictions[i], labels[i])  # call the function that has metric as name
                )
        self.losses[self.phase].append(loss)
        
    def set_phase(self, phase):
        self.phase = phase

    def save_epoch(self):
        for metric in self.metrics:
            for phase in self.metrics[metric]:
                self.writer.add_scalar("Mean " + metric + "/" + phase, np.mean(self.metrics[metric][phase]), self.epoch)
        for phase in self.losses:
            self.writer.add_scalar("Loss/" + phase, np.mean(self.losses[phase]), self.epoch)
        self.epoch += 1
        self._reset()

    def get_metric(self, metric, phase):
        if metric not in self.metrics:
            raise ValueError('metric not in ' + str(self.metrics.keys()))
        if phase not in ['train', 'valid']:
            raise ValueError('phase not in {train, valid}')
        return np.mean(self.metrics[metric][phase])

    def close(self):
        self.writer.close()
        

# TODO: include it in Hublot
def report(model, dataset, output_directory):
    data = DataLoader(dataset)
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
        for key in metrics:
            print(key, metrics[key])
        json.dump(metrics, open(output_directory+"/metrics.json", 'w'))