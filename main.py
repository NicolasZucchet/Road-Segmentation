#!/usr/bin/env python3


# TODO add GPU support

from src.images import load_train_data, overlays, save_all, MirroredRandomRotation
from src.data import RoadSegmentationDataset
from src.model import UNet
import numpy as np
import time, json
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# Experiment variables
UNET_INPUT_SIZE = 256
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 0
IN_CHANNELS = 3
N_CLASSES = 1
NAME = "TEST"
SAVE = True
MODEL_SAVE = "experiment_results/TEST_200423_122147/model.pt"


rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
transform_train = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    MirroredRandomRotation(15),
    transforms.RandomResizedCrop(UNET_INPUT_SIZE, scale=(0.3,1)),
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
if MODEL_SAVE is not None:
    model.load_state_dict(torch.load(MODEL_SAVE))
dataset_train = RoadSegmentationDataset('./data/training', indices=slice(70), train=True, transform=transform_train)
dataset_valid = RoadSegmentationDataset('./data/training', indices=slice(70, 100), train=True, transform=transform_test)
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

output_directory = "experiment_results/" + NAME + "_" + time.strftime("%y%m%d_%H%M%S", time.localtime())
if SAVE:
    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")
    os.makedirs(output_directory)
    print("\nExperiment results will be stored in ./"+output_directory)


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
            running_total += preds.size(0) * preds.size(1) * preds.size(2) * preds.size(3)

        epoch_acc = running_corrects.double() / float(running_total)
        print('\n\t({}) Accuracy: {:.4f}'.format(phase, epoch_acc))
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc

time_elapsed = time.time() - since
print('\nTrained in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

if SAVE:
    # save model
    print("Saving model...", end="")
    torch.save(model.state_dict(), output_directory+"/model.pt")
    print("\rModel saved   ")

def intersection_over_union(preds, gts):
    preds = (preds > 0.5).astype(np.bool)
    gts = gts.astype(np.bool)
    intersection = preds*gts
    union = preds+gts
    return np.sum(intersection)/float(np.sum(union))

def accuracy(preds, gts):
    preds = (preds > 0.5).astype(np.bool)
    gts = gts.astype(np.bool)
    n = gts.shape[0] * gts.shape[1] * gts.shape[2] * gts.shape[3]
    return np.sum(preds*gts)/float(n)

def report(model, dataset):
    data = DataLoader(dataset, batch_size=BATCH_SIZE)
    model.eval()  # set model in eval mode
    imgs, preds = [], []
    if dataset.train:
        gts = []
    print("Running...")
    for i, e in enumerate(data):
        print("\r\tBatch {}/{}".format(i+1, len(data)), end="")
        outputs = model(e['images'])  # forward pass
        outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
        # images/outputs have shape [batch_size, n_channels, height, width]
        # transform them into list of [height, width, n_channels
        preds += list(outputs.permute(0, 2, 3, 1).data.numpy())
        imgs += list(e['raw_images'].permute(0, 2, 3, 1).data.numpy())  # use raw images, which are images before normalization
        if dataset.train:
            gts += list(e['labels'].permute(0, 2, 3, 1).data.numpy())  # use raw images, which are images before normalization

    preds, imgs = np.array(preds), np.array(imgs)
    if dataset.train:
        gts = np.array(gts)

    results = overlays(imgs, preds, alpha = 0.4, binarize=True)
    if SAVE:
        save_all(results, output_directory+"/images")

    if dataset.train:
        metrics = {}
        metrics ['IOU'] = intersection_over_union(preds, gts)
        metrics['Accuracy'] = accuracy(preds, gts)
        for key in metrics:
            print(key, metrics[key])
        if SAVE:
            json.dump(metrics, open(output_directory+"/metrics.json", 'w'))

report(model, dataset_valid)
