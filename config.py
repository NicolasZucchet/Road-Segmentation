"""
This file contains parameters that can not be controled through command lines. Some of them
can be modified, others can't.
The ones that can be modified by the user are marked with XXX.
"""

from torchvision import transforms
from src.images import MirroredRandomRotation
import numpy as np
import torch


# Parameters for UNet
IN_CHANNELS = 3
N_CLASSES = 1
INPUT_SIZE = 400  # XXX


# Use GPU if possible
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)


# Parameters related to data augmentation
MIN_SCALE = 0.5  # XXX
MAX_SCALE = 1  # XXX
ROTATION_MAX_ANGLE = 10  # XXX
JITTER_BRIGHTNESS = 0.2  # XXX
JITTER_CONTRAST = 0.3  # XXX
JITTER_SATURATION = 0.1  # XXX
JITTER_HUE = 0.05  # XXX
# Transformation for training samples
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
TRANSFORM_TRAIN = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    MirroredRandomRotation(ROTATION_MAX_ANGLE),
    transforms.ColorJitter(
        brightness=JITTER_BRIGHTNESS, 
        contrast=JITTER_CONTRAST,
        saturation=JITTER_SATURATION, 
        hue=JITTER_HUE),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(MIN_SCALE, MAX_SCALE)),
        # take a patch of size scale*input_size, and resize it to INPUT_SIZE
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
]  # XXX
TRANSFORM_TEST = [
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
]  # XXX


# Random seed for numpy (that will be used to separate training and validation samples)
RANDOM_SEED = 42


# Parameters related to the GoogleMaps dataset
# Name of the tasks that should be used as training and validation tasks
TRAIN_TASKS = [
    'Rome', 'London', 'Vancouver', 'CapeTown', 'Tokyo', 
    'Stockholm', 'BuenosAires', 'Dubai', 'Canada_Countryside', 'Zurich', 
    'NewYork', 'Marakkesh', 'Auckland', 'Sidney', 'MexicoCity',
    'Mexico_Countryside', 'NewDelhi', 'Sweden_Countryside', 'Shanghai', 'Riyadh', 
    'Switzerland_Countryside', 'Teheran', 'Egypt_Countryside', 'France_Countryside', 'Italy_Countryside', 
    'Morocco_Countryside'
]
VAL_TASKS = [
    'Seattle', 'Miami', 'GB_Countryside', 'SanFrancisco', 'Paris', 
    'Chicago', 'Bankok', 'SouthAfrica_Countryside', 'RioDeJaneiro', 'Cairo'
]


# Number of samples in each task
N_SAMPLES_PER_TASK = 100