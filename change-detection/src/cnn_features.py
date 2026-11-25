"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import time

import numpy as np
import torch
import torchvision

from preprocess import shrink_to_width
from align import warp

vgg = None

# Loads VGG16 up to layer relu3_3 (downloads ~528 MB the first time).
def load_vgg():
    global vgg
    if vgg is None:
        torch.set_grad_enabled(False)
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        vgg = model.features[:16]
        vgg.eval()
    return vgg

# Feature maps of a BGR image
def extract_features(im):
    im = shrink_to_width(im, 400) / 255
    height, width, _ = im.shape
    rgb = np.zeros((1, 3, height, width))
    rgb[0, 0] = (im[:, :, 2] - 0.485) / 0.229   # red
    rgb[0, 1] = (im[:, :, 1] - 0.456) / 0.224   # green
    rgb[0, 2] = (im[:, :, 0] - 0.406) / 0.225   # blue
    features = load_vgg()(torch.tensor(rgb, dtype=torch.float32))
    return features[0].numpy()

# the cosine similarity of the feature vectors at each location
def cnn_difference(before, after):
    height, width, _ = before.shape
    features1 = extract_features(before)
    features2 = extract_features(after)

    dot = np.sum(features1 * features2, axis=0)
    norms = np.sqrt(np.sum(features1 ** 2, axis=0)) * np.sqrt(np.sum(features2 ** 2, axis=0))
    distance = 1 - dot / (norms + 10**-8)
    distance[norms < 10**-8] = 0

    small_height, small_width = distance.shape
    scale = np.array([[small_width / width, 0.0, 0.0], [0.0, small_height / height, 0.0], [0.0, 0.0, 1.0]])
    distance, _ = warp(distance, scale, height, width)
    return np.clip(distance, 0, 1)

if __name__ == "__main__":
    im = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)
    start = time.time()
    change = cnn_difference(im, im)
    print("CNN time (s):", round(time.time() - start, 2))
    print("Max difference for the same image (should be about 0):", round(change.max(), 4))
