# Meta-learning for road segmentation

This repo contains tools for classical supervised learning and meta Learning for the task of aerial road segmentation.

## Setup

### In general 
`pip install -r requirements.txt`

### On Leonhard

`module load python_gpu/3.6.4 hdf5 eth_proxy cudnn/7.2`<br/>
`pip install virtualenvwrapper`<br/>
`source $HOME/.local/bin/virtualenvwrapper.sh`<br/>
`mkvirtualenv env`<br/>
`pip install -r requirements.txt`


## Structure of the repo

### `data`

It contains two different datasets:
- GoogleMaps: a homemade dataset that contains 100 labelled images for 35 different regions of the world;
- CIL: the dataset given for the CIL project.

### `src`

This folder contains all the core elements:
- `data.py`: tools to load the data in an easy to use way;
- `images.py`: tools to manipulate (load, save, create overlays...) images;
- `loss.py`: definition of the loss (BCE with Dice penalization);
- `metrics.py`: tools to monitor the training of the network, storing them so that they can be seen in Tensorboard;
- `model.py`: definition of the model (vanilla U-Net).

### `supervised.py`

Script that trains the network in the classical supervised fashion. It can be used on CIL and GoogleMaps datasets. Note that it should be used through command line.

### `meta.py`

Script that trains the network in a meta learning fashion. It uses the REPTILE algorithm and can only deal with GoogleMaps data set. It should also be used through command line.

### `config.py`

Contains all the parameters used in the repo that are not entered through command line. Some parameters can be modified, we marked them with `# XXX`.

### `experiment_results`

This folder is created when the first experiment is launched. It contains all the experiment results (model weights, Tensorboard files...).


## A training example

We train our network in the following way:
1. on GoogleMaps data, in the classical way: `python supervised.py --dataset GoogleMaps --name first_step --epochs 100 --n-train 50`;
2. on GoogleMaps data, in the meta learning way: `python meta.py --name second_step --epochs 100 --save experiment_results/PATH_TO_MODEL_OF_STEP_1/model.pt`;
3. on CIL data, in the classical way: `python supervised.py --dataset CIL --name third_step --epochs 50  --save experiment_results/PATH_TO_MODEL_OF_STEP_2/model.pt`.

The model is then ready to be applied to the testing set: **TODO**

Some metrics are saved during the training and can be viewed within tensorboard. To launch it, in `experiment_results` folder, run `tensorboard --logdir . --port 16006`. <br/>
If Tensorboard is running on your computer, you can access it through `http://localhost:16006` in a web navigator. <br/>
If it is running on Leonhard, you first need to execute `ssh -L 6006:127.0.0.1:16006 USERNAME@login.leonhard.ethz.ch` on your local machine, and then go to `http://localhost:6006`.
