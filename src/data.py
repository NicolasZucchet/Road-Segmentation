from images import load_train_data, load
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms



class RoadSegmentationDataset(Dataset):
    """Road segmentation dataset."""

    def __init__(self, root_dir, indices=None, train=True, transform=None):
        """
        Args:
            root_dir (String): Directory with all the data.
            indices (int list): list of the indices of images to consider
            train (bool): If it is a training dataset or a testing one.
            transform (list of transforms): List of transformations to be applied on a sample.
                Last transformation must be an instance of `transforms.Normalize`
        """
        self.root_dir = root_dir
        self.images, self.labels = None, None
        self.train = train
        if self.train:
            self.images, self.labels = load_train_data(self.root_dir, indices=indices)
            assert len(self.images) == len(self.labels)
        else:
            self.images = load(self.root_dir, indices=indices)
        self.transform = transform
        if self.transform is not None:
            assert type(self.transform[-1]) == transforms.Normalize
            self.data_transform = transforms.Compose(self.transform)
            self.label_transform = transforms.Compose(self.transform[:-1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get corresponding images
        images = self.images[idx]

        if self.train:
            labels = self.labels[idx]  # only in train mode

        # apply specific transformation for images and labels if in train mode
            # use trick presented in https://github.com/pytorch/vision/issues/9#issuecomment-383110707
        if self.train:
            seed = random.randint(0,2**32)
            random.seed(seed)
            images = self.data_transform(images)
            random.seed(seed)
            labels = self.label_transform(labels)

            # to ensure label in {0, 1}
            labels = (labels > 0.5).float()

            # one hot encode labels
            #one_hot = torch.FloatTensor(2, labels.size(1), labels.size(2)).zero_()
            #labels = one_hot.scatter_(0, labels, 1)

        # define corresponding sample
        if self.train:
            sample = {'images': images, 'labels': labels}
        else:
            sample = {'images': images}

        return sample
