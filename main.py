#!/usr/bin/env python3

# add source to the Python path
import os, sys
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)


# TODO add rotation in transforms (+ miroring to avoid null pixels)
# TODO add testing phase (overlay image and prediction, save in unique directory)
# TODO add GPU support

from images import load_train_data
from data import RoadSegmentationDataset
from model import UNet
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

UNET_INPUT_SIZE = 256
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 1
IN_CHANNELS = 3
N_CLASSES = 1


rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
transform_train = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(UNET_INPUT_SIZE, scale=(0.2,1)),  
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
dataset_train = RoadSegmentationDataset('./data/training', indices=slice(70), train=True, transform=transform_train)
dataset_valid = RoadSegmentationDataset('./data/training', indices=slice(70, 100), train=True, transform=transform_train)
dataset_test = RoadSegmentationDataset('./data/test', train=False, transform=transform_test)
data = {
    "train": DataLoader(dataset_train, batch_size=BATCH_SIZE),
    "valid": DataLoader(dataset_valid, batch_size=BATCH_SIZE),
    "test": DataLoader(dataset_test, batch_size=BATCH_SIZE)
}

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

since = time.time()
best_acc = 0.0

for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch+1, EPOCHS))

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        # to keep trace of prediction results    
        running_corrects = 0
        running_total = 0

        # Iterate over data and labels (minibatches), by default, for one epoch.
        for i, e in enumerate(data[phase]):
            print("\r\tBatch {}/{}".format(i+1, len(data[phase])), end="")
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(e['images'])  # forward pass
                outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
                preds = (outputs > 0.5).float()
                loss = criterion(outputs, e['labels'])

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_corrects += torch.sum(preds == e['labels'].data)
            running_total += preds.size(0) * preds.size(1) * preds.size(2)

        epoch_acc = running_corrects.double() / running_total.float()
        print('({}) Accuracy: {:.4f}'.format(epoch_acc))
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc

time_elapsed = time.time() - since
print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))