from .images import load_train_data, load, load_train_data_directories, load_directories
import torch
import os
from torch.utils.data import Dataset
import random
from torchvision import transforms
import numpy as np


class RoadSegmentationDataset(Dataset):
    """Road segmentation dataset."""

    def __init__(self, root_dir, indices=None, train=True, transform=None, device=None, subtasks=False, verbose=True):
        """
        Args:
            root_dir (String): Directory with all the data.
            indices (int list): list of the indices of images to consider
            train (bool): If it is a training dataset or a testing one (ie no label).
            transform (list of transforms): List of transformations to be applied on a sample.
                Last transformation must be an instance of `transforms.Normalize`
            device (torch.device): device to use
            subtasks (bool): weither the data in root_dir contains sub tasks
            verbose (bool): To display some details
        """
        self.root_dir = root_dir
        self.images, self.labels = None, None
        self.train = train

        if subtasks is True:
            self.subtasks = True
            subdirectories = [
                os.path.join(root_dir, x) for x in os.listdir(root_dir) 
                if "." not in x
            ]
        elif subtasks is not False:
            self.subtasks = True
            subdirectories = [
                os.path.join(root_dir, x) for x in subtasks
            ]
        else:
            self.subtasks = False

        if self.train:
            if not self.subtasks:
                self.images, self.labels = load_train_data(self.root_dir, indices=indices, verbose=verbose)
            else:
                self.images, self.labels = load_train_data_directories(subdirectories, indices=indices, verbose=verbose)
            assert len(self.images) == len(self.labels)
        else:
            if not self.subtasks:
                self.images = load(self.root_dir, indices=indices, verbose=verbose)
            else:
                self.images = load_directories(subdirectories, indices=indices, verbose=verbose)
        self.transform = transform
        if self.transform is not None:
            assert type(self.transform[-1]) == transforms.Normalize
            self.data_transform = transforms.Compose(self.transform)
            # remove normalization
            self.label_transform = transforms.Compose(self.transform[:-1])
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get corresponding images
        raw_images = self.images[idx]

        if self.train:
            labels = self.labels[idx]  # only in train mode

        # apply specific transformation for images and labels if in train mode
            # use trick presented in https://github.com/pytorch/vision/issues/9#issuecomment-383110707
        seed = random.randint(0,2**32)
        random.seed(seed)
        images = self.data_transform(raw_images)
        random.seed(seed)
        raw_images = self.label_transform(raw_images)
        
        if self.train:
            random.seed(seed)
            labels = self.label_transform(labels).float()
            # to ensure label in {0, 1}
            labels = (labels > 0.5).float()

        # define corresponding sample
        if self.train:
            sample = {
                'images': images.to(device=self.device), 
                'raw_images': raw_images.to(device=self.device), 
                'labels': labels.to(device=self.device)
            }
        else:
            sample = {
                'raw_images': raw_images.to(device=self.device), 
                'images': images.to(device=self.device)
            }

        return sample


class RoadSegmentationTask:
    """
    Class containing train and val dataloaders for a given task (i.e. one of the GoogleMaps dataset)
    """

    def __init__(self, root_dir, train_indices, val_indices, train_transform=None, val_transform=None, device=None, verbose=True):
        self.root_dir = root_dir
        self.train_data = RoadSegmentationDataset(
            root_dir, indices=train_indices, transform=train_transform, device=device, verbose=verbose)
        self.val_data = RoadSegmentationDataset(
            root_dir, indices=val_indices, transform=val_transform, device=device, verbose=verbose)