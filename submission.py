import os, glob, re
import numpy as np
from src.model import Model
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import matplotlib.image as mpimg
import matplotlib as mpl
import config
import cv2


device = config.device


""" Parameters (## when should/can be modified)"""

# parameters for UNet
IN_CHANNELS = 3
N_CLASSES = 1

# path to the model weights to load
SAVE = 'model_3.pt'  ##
# loads model and weigths
model = Model('UNet', IN_CHANNELS, N_CLASSES, device=device)
model.load_weights(path=SAVE)

# post processing model
postprocessing_weights = "postprocessing.tar"
postprocessing_model = UNet(IN_CHANNELS+1, N_CLASSES).to(device)
postprocessing_model.load_weights(path=postprocessing_weights)

checkpoint = torch.load("models/postprocessing_400.tar")
postprocessing_model.load_state_dict(checkpoint['model_state_dict'])

# path to the directory where test images are stored
TEST_IMAGES_PATH = 'data/CIL/test'  ##

# size of images
LENGTH = 608
# size of batches to take in input image
OUTPUT =  400
# number of patches to have (horizontally and vertically)
N = 2  ##
if N == 1:
    GAP =  0
else:
    GAP = int((LENGTH - OUTPUT)/(N-1))
assert GAP < OUTPUT, 'N should be bigger'

# name of the submission
SUBMISSION_NAME = 'submission.csv'  ##

# Transformation to apply on images
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
transform_test = [
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
]
transform = transforms.Compose(transform_test)


""" For postprocessing """
def postprocess(mask, image, n):
    postprocessing_model.eval()
    mask = torch.cat([mask, image], dim=1)
    output = postprocessing_model(mask.cuda())
    if n != 0:
        for i in range(n):
            output = torch.cat([torch.sigmoid(output), image.cuda()], dim=1)
            output = postprocessing_model(output)
    return (torch.sigmoid(output)>0.5).float()


""" Functions related to conversion into submission format """

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    foreground_threshold = 0.25
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(im, file_name):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", file_name).group(0))
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

""" Functions to create batches from an image to do prediction on them, and then mix them """

def image_to_gridbatch(img):
    """Transforms an image into a batch of patches"""
    images = []
    for i in range(N):
        for j in range(N):
            images.append(img[0, :, i*GAP:i*GAP+OUTPUT, j*GAP:j*GAP+OUTPUT])
    return torch.stack(images)

def gridbatch_to_image(gridbatch):
    """Gather a batch of patches to form a single image"""
    img = torch.zeros([LENGTH, LENGTH, 1]).to(device)
    count = torch.zeros([LENGTH, LENGTH]).to(device)
    for i in range(N):
        for j in range(N):
            ind = i*(N) + j
            img[i*GAP:i*GAP+OUTPUT, j*GAP:j*GAP+OUTPUT, :] += gridbatch[ind]
            count[i*GAP:i*GAP+OUTPUT, j*GAP:j*GAP+OUTPUT] += 1.0
    img[:,:,0] /= count
    return img


""" Functions for overlaying predictions on raw images """

def img_float_to_uint8(img):
    return (img * 255).round().astype(np.uint8)

def img_binarize(img, threshold=0.25):
    return (img >= threshold).astype(np.float)

def overlays(imgs, masks, alpha=0.95, binarize=False):
    """Add the masks on top of the images with red transparency
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
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    if len(images.shape) == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)
    cmap = mpl.rcParams.get("image.cmap")

    for n in range(images.shape[0]):
        mpimg.imsave(os.path.join(directory, format_.format(n + 1)), images[n], cmap=cmap)


""" Main function """

def generate(model, directory):
    """Generates predictions, outputs them for submission and for visual inspection"""
    model.eval()
    predictions, raw_images = [], []

    submission_filename = SUBMISSION_NAME
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')

    paths = sorted(glob.glob(os.path.join(directory, '*.png')))
    for i, file_path in enumerate(paths):
        print(f'\r{i}/{len(paths)}', end='')
        raw_image = Image.open(file_path)
        image = transform(raw_image).to(device)
        image.unsqueeze_(0)
        images = F.interpolate(image_to_gridbatch(image), size=256)

        with torch.set_grad_enabled(False):
            outputs = model(images)  # forward pass
            outputs = torch.sigmoid(outputs)  # apply sigmoid to restrain within [0, 1]
             # Post-processing
            preds = F.interpolate(preds, size=OUTPUT)
            image = F.interpolate(images, size=OUTPUT)
            preds = postprocess(preds.cuda(), image.cuda(), 0)
            preds = torch.FloatTensor([cv2.blur(p.squeeze().cpu().detach().numpy(), (25,25)) for p in preds]).unsqueeze(3).permute(0,3,1,2).to(device)
            preds = (preds > 0.5).float()
            preds = preds.permute(0, 2, 3, 1) 
            pred = gridbatch_to_image(preds).cpu().numpy()
        
        predictions.append(list(pred))

        with open(submission_filename, 'a') as f:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(pred, file_path))
        
        raw_image = np.array(raw_image)
        raw_images.append(list(raw_image))

    raw_images = 255-np.array(raw_images)
    predictions = np.array(predictions)
    results = overlays(raw_images, predictions, alpha=0.4, binarize=True)
    save_all(results, "images")
    save_all(predictions, "predictions")


generate(model, TEST_IMAGES_PATH)
