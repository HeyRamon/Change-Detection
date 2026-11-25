"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import cv2
import numpy as np

from preprocess import gaussian_filter, compute_gradient
from segment import dilate

# Grayscale difference scaled from 0 to 1
def gray_difference(before_gray, after_gray):
    return np.clip(np.abs(before_gray - after_gray) / 255, 0, 1)

# Color difference
def color_difference(before, after):
    before = before.astype(float)
    after = after.astype(float)
    distance = np.sqrt((before[:, :, 0] - after[:, :, 0]) ** 2
                       + (before[:, :, 1] - after[:, :, 1]) ** 2
                       + (before[:, :, 2] - after[:, :, 2]) ** 2)
    return np.clip(distance / 255, 0, 1)

# Difference of gradient magnitudes
def sobel_difference(before_gray, after_gray):
    gradient_x, gradient_y = compute_gradient(before_gray)
    magnitude1 = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_x, gradient_y = compute_gradient(after_gray)
    magnitude2 = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    magnitude1 = gaussian_filter(magnitude1 / magnitude1.mean(), 2)
    magnitude2 = gaussian_filter(magnitude2 / magnitude2.mean(), 2)
    return np.clip(np.abs(magnitude1 - magnitude2) / 3, 0, 1)

# Canny edges that have no edge within 3 px in the other image are blurred into a map.
def canny_difference(before_gray, after_gray):
    edges1 = cv2.Canny(before_gray.astype(np.uint8), 50, 150) > 0
    edges2 = cv2.Canny(after_gray.astype(np.uint8), 50, 150) > 0

    near_edges1 = dilate(edges1, 7)
    near_edges2 = dilate(edges2, 7)
    removed = edges1 & (near_edges2 == False)
    added = edges2 & (near_edges1 == False)
    unmatched = removed | added

    # a blurred 1-pixel line has a peak of about 1/7.5, so this makes a line = 1
    density = gaussian_filter(unmatched.astype(float), 3) * 7.5
    return np.clip(density, 0, 1)

# Change map (0-1) with the chosen method
def compute_change_map(method, before, before_gray, after, after_gray, valid):
    if method == "gray":
        change = gray_difference(before_gray, after_gray)
    elif method == "color":
        change = color_difference(before, after)
    elif method == "sobel":
        change = sobel_difference(before_gray, after_gray)
    elif method == "canny":
        change = canny_difference(before_gray, after_gray)
    else:
        # PyTorch is only loaded when the CNN method is used
        from cnn_features import cnn_difference
        change = cnn_difference(before, after)
    change[valid == False] = 0
    return change