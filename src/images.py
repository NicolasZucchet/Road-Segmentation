import glob
import os
import sys

import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import rotate

FOREGROUND_THRESHOLD = .25
PIXEL_DEPTH = 255


"""
IMAGES
    This modules gather a handful of utility functions to manipulate images and masks.
    Most of the functions are defined for batches of images (4D tensors).

Slightly modified version of https://github.com/aschneuw/road-segmentation-unet/src/images.py
"""


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
    return (img > threshold).astype(np.float)


def load(directory, indices=None):
    """Extract the images in `directory` into a tensor [num_images, height, width(, channels)]
    
    Args:
        directory (String): path to the directery
        indices (slice): files to keep
        
    Returns:
        list of 2D or 3D images
    """

    print('Loading images from {}...'.format(directory.split("/")[-1]))
    images = []
    paths = sorted(glob.glob(os.path.join(directory, '*.png')))
    if indices is not None:
        paths = paths[indices]
    for i, file_path in enumerate(paths):
        print('\rImage {}/{}'.format(i, len(paths)), end='')
        images.append(Image.open(file_path))
    print("\rLoaded {} images ".format(len(images), directory))
    return images


def load_train_data(directory, indices=None):
    """Load images and labels from `directory`

    Args:
        directory (String): path to the directory, must contain subfolders `images` and `groundtruth`
        indices (slice): files to keep
    returns:
        images: [num_images, img_height, img_width, num_channel]
        labels: [num_images, img_height, img_width]
    """
    train_data_dir = os.path.abspath(os.path.join(directory, 'images/'))
    train_labels_dir = os.path.abspath(os.path.join(directory, 'groundtruth/'))

    train_images = load(train_data_dir, indices=indices)
    train_groundtruth = load(train_labels_dir, indices=indices)

    return train_images, train_groundtruth


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



# TODO review....


def mirror_border(images, n):
    """mirrors border n border pixels on each side and corner:
        4D [num_images, image_height, image_width, num_channel]
        or 3D [num_images, image_height, image_width]
    returns:
        4D input: [num_patches, patch_size, patch_size, num_channel]
        3D input: [num_patches, patch_size, patch_size]
    """
    has_channels = (len(images.shape) == 4)
    if has_channels:
        return np.pad(images, ((0, 0), (n, n), (n, n), (0, 0)), "symmetric")
    else:
        return np.pad(images, ((0, 0), (n, n), (n, n)), "symmetric")


def overlap_pred_true(pred, true):
    num_images, im_height, im_width = pred.shape
    true_mask = img_float_to_uint8(true)
    pred_mask = img_float_to_uint8(pred)

    overlapped_mask = np.zeros((num_images, im_height, im_width, 3), dtype=np.uint8)
    overlapped_mask[:, :, :, 0] = pred_mask
    overlapped_mask[:, :, :, 1] = true_mask
    overlapped_mask[:, :, :, 2] = 0

    return overlapped_mask


def overlapp_error(pred, true):
    num_images, im_height, im_width = pred.shape
    true_mask = img_float_to_uint8(true).astype("bool", copy=False)
    pred_mask = img_float_to_uint8(pred).astype("bool", copy=False)
    error = np.logical_xor(true_mask, pred_mask)
    np.logical_not(error, out=error)
    error = img_float_to_uint8(error * 1)

    error_mask = np.zeros((num_images, im_height, im_width, 3), dtype=np.uint8)
    error_mask[:, :, :, 0] = error
    error_mask[:, :, :, 1] = error
    error_mask[:, :, :, 2] = error

    return error_mask


def rotate_imgs(imgs, angle):
    """safeguard to avoid useless rotation by 0"""
    if angle == 0:
        return imgs
    return rotate(imgs, angle=angle, axes=(1, 2), order=0)


def expand_and_rotate(imgs, angles, offset=0):
    """rotate some images by an angle, mirror image for missing part and expanding to output_size
        4D [num_images, image_height, image_width, num_channel]
        or 3D [num_images, image_height, image_width]
    angles: list of angle to rotate
    output_size: new size of image
    returns:
        4D input: [num_images * num_angles, output_size, output_size, num_channel]
        3D input: [num_images * num_angles, output_size, output_size]
    """

    has_channels = (len(imgs.shape) == 4)
    if not has_channels:
        imgs = np.expand_dims(imgs, -1)

    batch_size, height, width, num_channel = imgs.shape
    assert height == width

    output_size = height + 2 * offset
    padding = int(np.ceil(height * (np.sqrt(2) - 1) / 2)) + int(np.ceil(offset / np.sqrt(2)))

    print("Applying rotations: {} degrees... ".format(", ".join([str(a) for a in angles])))
    imgs = mirror_border(imgs, padding)
    rotated_imgs = np.zeros((batch_size * len(angles), output_size, output_size, num_channel))
    for i, angle in enumerate(angles):
        rotated_imgs[i * batch_size:(i + 1) * batch_size] = crop_imgs(rotate_imgs(imgs, angle), output_size)
    print("Done")

    if not has_channels:
        rotated_imgs = np.squeeze(rotated_imgs, -1)

    return rotated_imgs


def crop_imgs(imgs, crop_size):
    """
    imgs:
        3D or 4D images batch
    crop_size:
        width and height of the input
    """
    batch_size, height, width = imgs.shape[:3]
    assert height == width and height >= crop_size
    assert crop_size % 2 == 0
    half_crop = int(crop_size / 2)
    center = int(height / 2)

    has_channels = (len(imgs.shape) == 4)
    if has_channels:
        croped = imgs[:, center - half_crop:center + half_crop, center - half_crop:center + half_crop, :]
    else:
        croped = imgs[:, center - half_crop:center + half_crop, center - half_crop:center + half_crop]

    return croped


def image_augmentation_ensemble(imgs):
    """create ensemble of images to be predicted

    imgs: 4D images batch [num_images, height, width, channels]
    returns:  4D images batch [6 * num_images, height, width, channels]
    """
    num_imgs = imgs.shape[0]
    augmented_imgs = np.zeros((num_imgs * 6,) + imgs.shape[1:])

    # originals
    augmented_imgs[:num_imgs] = imgs

    # horizontal and vertical flip
    augmented_imgs[num_imgs:2 * num_imgs] = np.flip(imgs, axis=2)
    augmented_imgs[2 * num_imgs:3 * num_imgs] = np.flip(imgs, axis=1)

    # rotated images
    for i, k in enumerate([1, 2, 3]):
        augmented_imgs[(3 + i) * num_imgs:(4 + i) * num_imgs] = np.rot90(imgs, k=k, axes=(1, 2))

    return augmented_imgs


def invert_image_augmentation_ensemble(masks):
    """assemble masks of prediction images created by `image_augmentation_ensemble`

    masks: 3D masks batch [6 * num_images, height, width]
    returns: 3D masks batch [num_images, height, width]
    """
    assert masks.shape[0] % 6 == 0
    num_imgs = int(masks.shape[0] / 6)

    result = masks[:num_imgs]

    result += np.flip(masks[num_imgs:2 * num_imgs], axis=2)
    result += np.flip(masks[2 * num_imgs:3 * num_imgs], axis=1)

    # rotated images
    for i, k in enumerate([-1, -2, -3]):
        result += np.rot90(masks[(3 + i) * num_imgs:(4 + i) * num_imgs], k=k, axes=(1, 2))

    return result / 6
