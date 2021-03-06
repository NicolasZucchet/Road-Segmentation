"""
This modules gather a handful of utility functions to manipulate images and masks.
Most of the functions are defined for batches of images (4D tensors).

Custom version of https://github.com/aschneuw/road-segmentation-unet/src/images.py
"""


import glob
import os
import sys

import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import random 
from PIL import Image
from scipy.ndimage.interpolation import rotate

FOREGROUND_THRESHOLD = .25
PIXEL_DEPTH = 255


def img_float_to_uint8(img):
    """Transform an array of float images into uint8 images
    
    Args:
        img (float np.array): pixel values between 0 and 1
        
    Returns:
        uint8 np.array
    """
    return (img * PIXEL_DEPTH).round().astype(np.uint8)


def img_binarize(img, threshold=0.5):
    """ Transform a float image into a binary image (0 or 1)
    
    Args:
        img (float np.array): pixels values between 0 and 1
        threshold (float): threshold to separate 0s and 1s
    """
    return (img >= threshold).astype(np.float)


def load(directory, indices=None):
    """Extract image paths in `directory`
    
    Args:
        directory (String): path to the directery
        indices (slice): files to keep
        
    Returns:
        paths: list of file paths
    """
    paths = sorted(glob.glob(os.path.join(directory, '*.png')))
    if indices is not None:
        paths = [paths[i] for i in indices]
    return paths


def load_directories(directories, indices=None):
    """Version of load for several directories"""
    paths = []
    for directory in directories:
        paths += load(directory, indices=indices)
    return paths


def load_train_data(directory, indices=None):
    """Load image and label paths from `directory`

    Args:
        directory (String): path to the directory, must contain subfolders `images` and `groundtruth`
        indices (slice): files to keep
    returns:
        image_paths: list of paths to images
        label_paths: list of paths to labels
    """
    train_images_dir = os.path.abspath(os.path.join(directory, 'images/'))
    train_labels_dir = os.path.abspath(os.path.join(directory, 'groundtruth/'))

    image_paths = load(train_images_dir, indices=indices)
    label_paths = load(train_labels_dir, indices=indices)

    return image_paths, label_paths


def load_train_data_directories(directories, indices=None):
    """Version of load_train_data for several directories
    """
    image_paths, label_paths = [], []
    for directory in directories:
        train_images_dir = os.path.abspath(os.path.join(directory, 'images/'))
        train_labels_dir = os.path.abspath(os.path.join(directory, 'groundtruth/'))

        image_paths += load(train_images_dir, indices=indices)
        label_paths += load(train_labels_dir, indices=indices)

    return image_paths, label_paths


def overlays(imgs, masks, alpha=0.95, binarize=False):
    """Add the masks on top of the images with red transparency

    Args:
        imgs (float np.array):
            shape: [num_images, height, width, num_channel]
        masks (float np.array):
            array of masks
            shape: [num_images, height, width, 1]
        alpha (float):
            between 0 and 1, alpha channel value for the mask
        binarize(bool):
            if the mask should be consider transformed in {0, 1} (instead of [0, 1])
    
    Returns:
        np.array, shape: [num_images, height, width, num_channel]
    """
    num_images, im_height, im_width, num_channel = imgs.shape
    assert num_channel == 3, 'PImages should be RGB images'

    if binarize:
        masks = img_binarize(masks)

    imgs = img_float_to_uint8(imgs)
    masks = img_float_to_uint8(masks.squeeze())
    masks_red = np.zeros((num_images, im_height, im_width, 4), dtype=np.uint8)
    masks_red[:, :, :, 0] = 255
    masks_red[:, :, :, 3] = masks * alpha

    results = np.zeros((num_images, im_width, im_height, 4), dtype=np.uint8)
    for i in range(num_images):
        x = Image.fromarray(imgs[i]).convert('RGBA')
        y = Image.fromarray(masks_red[i])
        results[i] = np.array(Image.alpha_composite(x, y))

    return results


def save_all(images, directory, format_="images_{:03d}.png", greyscale=False):
    """Save the `images` in the `directory`

    Args:
        images (uint8 np.array): 
            3D or 4D tensor of images (with or without channels)
        directory (String): target directory path
        format_ (String): image naming with a placeholder for a integer index
        greyscale (bool): saved image in greyscale or not
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    if len(images.shape) == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)

    if greyscale:
        cmap = "gray"
    else:
        cmap = mpl.rcParams.get("image.cmap")

    for n in range(images.shape[0]):
        mpimg.imsave(os.path.join(directory, format_.format(n + 1)), images[n], cmap=cmap)


def crop_image(image, crop_size):
    """
    Crop centered image
    Args:
        imgs: 3D np.array (n_channels, height, width)
        crop_size (int): width and height of the input, must be even
    """
    _, height, width = image.shape
    assert height == width and height >= crop_size
    assert crop_size % 2 == 0
    half_crop = int(crop_size / 2)
    center = int(height / 2)
    return image[:, center-half_crop:center+half_crop, center-half_crop:center+half_crop]


class MirroredRandomRotation():
    """
    Random rotation with mirrored edges to fill in the blanks.

    Args:
        max_angle: in degree, angles will be sampled in [0, max_angle]
    """
    def __init__(self, max_angle):
        self.max_angle = max_angle
        self.angles = np.arange(0, max_angle, 1)

    def __call__(self, image):
        # image: n_channels, height, width (PIL image)
        # from PIL to np.array
        np_image = np.array(image)
        dim = len(np_image.shape)
        if dim == 2:
            np_image = np.expand_dims(np_image, -1) 
        np_image = np_image.transpose(2, 0, 1)
    
        _, height, width = np_image.shape
        assert height == width
        n = int(np.ceil(height * (np.sqrt(2) - 1) / 2))  # minimum padding s.t. no blank pixels after rotation
        angle = self.angles[random.randint(0, len(self.angles)-1)]  
            # choose one of the angles at random

        # transformations
        mirrored_image = np.pad(np_image, ((0, 0), (n, n), (n, n)), "symmetric")  # mirrors n border pixel
        rotated_image = rotate(mirrored_image, angle=angle, axes=(1, 2), order=1)  # rotate the images
        final_image = crop_image(rotated_image, height)  # crop images according to input size

        # back to PIL image
        final_image = final_image.transpose(1, 2, 0)
        if dim == 2:
            final_image = final_image[:, :, 0]

        return Image.fromarray(final_image)
        
