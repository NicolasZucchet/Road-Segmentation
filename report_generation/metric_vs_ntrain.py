import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.loss import BCEDicePenalizeBorderLoss
# Project imports
from src.data import RoadSegmentationDataset
from src.model import Model
from src.metrics import f1_score, accuracy, intersection_over_union
# General imports
import numpy as np
import time
import os
# Torch imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import sklearn.metrics
import config


os.chdir('..')


IN_CHANNELS = 3
N_CLASSES = 1
SAVE_META = 'model_2.pt'
SAVE_NORMAL = 'model_1.pt'

N_EPOCHS = 35
N_REPETITIONS = 1
# number of training samples
N_TRAINS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70]
# save in CSV format (to use when N_REPETITIONS is high, to avoid time limit)
SAVE_CSV = True
SAVE_PLOTS = False


device = config.device
transform_train, transform_test = config.TRANSFORM_TRAIN, config.TRANSFORM_TEST

def train(model, data):
    criterion = BCEDicePenalizeBorderLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(N_EPOCHS):
        # Iterate over data and labels (minibatches), by default, for one epoch.
        for e in data['train']:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(e['images'])  # forward pass
                loss = criterion(outputs, e['labels'])
                outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
                preds = (outputs > 0.5).float()
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()


def evaluate(model, data):
    results = {'IoU':[], 'F1':[], 'Acc':[]}
    model.eval()
    for e in data['val']:
        with torch.set_grad_enabled(False):
            outputs = model(e['images'])  # forward pass
            outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
            preds = (outputs > 0.5).float()

            preds = preds.cpu().numpy()
            labels = e['labels'].cpu().numpy()

            results['IoU'].append(intersection_over_union(preds,labels))
            results['F1'].append(f1_score(preds,labels))
            results['Acc'].append(accuracy(preds,labels))

    for key in results:
        results[key] = np.mean(results[key])
    return results


def generate_results(model):
    results = []

    for i in range(N_REPETITIONS):
        print(f'{i+1}/{N_REPETITIONS}')
        random_permutation = np.random.permutation(100)
        indices_val = random_permutation[-30:]
        dataset_val = RoadSegmentationDataset('./data/CIL/training', indices=indices_val, train=True, transform=transform_test, device=device)
        
        for n_train in N_TRAINS:
            print(f'\t{n_train}/70')
            t = time.time()
            indices_train = np.random.permutation(random_permutation[:70])
            _it = indices_train[:n_train]
            dataset_train = RoadSegmentationDataset('./data/CIL/training', indices=_it, train=True, transform=transform_train, device=device)
            datasets = {
                "train": dataset_train,
                "val": dataset_val
            }
            data = { key: DataLoader(datasets[key],  batch_size=10) for key in datasets.keys() }
            train(model, data)
            _res = evaluate(model, data)
            results.append([n_train, _res['IoU'], _res['F1'], _res['Acc']])
            print('\tTime', time.time()-t)

    return results

methods = ['Meta learning', 'Transfer learning', 'No learning']
metrics = ['mIoU', 'F1', 'Accuracy']
results = {meth:[] for meth in methods}
for i, save in enumerate([SAVE_META, SAVE_NORMAL, None]):
    model = Model('UNet', config.IN_CHANNELS, config.N_CLASSES, device=device)
    model.load_weights(path=save)
    results[methods[i]] = pd.DataFrame(generate_results(model), columns= ['Number of training samples from the task of interest']+metrics)
    
if SAVE_CSV:
    n = np.random.randint(200000)
    for method in results:
        results[method].to_csv(method+str(n)+'.csv')
    
if SAVE_PLOTS:
    for metric in metrics:
        with sns.axes_style("whitegrid"):
            for method in methods:
                print(results[method])
                ax = sns.lineplot(x="Number of training samples from the task of interest", y=metric, data=results[method], label=method)
            plt.legend(loc='lower right')
            plt.title(metric)
            plt.savefig(os.path.join('.', metric+'.png'), dpi=500)
            plt.clf()
            plt.close()