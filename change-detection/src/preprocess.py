"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import os

import cv2
import numpy as np

# Builds a 2D Gaussian kernel from a 1D Gaussian.
def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel

# 2D convolution
def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_out = cv2.filter2D(im.astype(float), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return im_out

# Gaussian smoothing.
def gaussian_filter(im, sigma):
    kernel_size = 2 * int(3 * sigma) + 1
    kernel = get_gaussian_kernel(kernel_size, sigma)
    return convolution(im, kernel)

# Returns the x and y gradients.
def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    return gradient_x, gradient_y

# Halves the image until its width is at most max_width.
def shrink_to_width(im, max_width):
    while im.shape[1] > max_width:
        im = gaussian_filter(im, 1.0)[::2, ::2]
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im

# Histogram of one 8-bit channel.
def histogram_equalization(channel):
    hist = np.zeros(256)
    for value in range(256):
        hist[value] = np.sum(channel == value)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    lookup = (cdf - cdf_min) / (channel.size - cdf_min) * 255
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    return lookup[channel]

# Equalizes only the V channel in HSV so colors are kept.
def equalize_color(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_hsv[:, :, 2] = histogram_equalization(im_hsv[:, :, 2])
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

# Returns the smoothed color image and grayscale image.
def preprocess(im, sigma=1.0, equalize=False):
    if equalize:
        im = equalize_color(im)
    im_color = np.clip(gaussian_filter(im, sigma), 0, 255).astype(np.uint8)
    im_gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY).astype(float)
    return im_color, im_gray

# Reads one column of data/pairs.csv (0 = name, 1 = scene, 2 = condition).
def read_pairs_csv(column=0):
    values = []
    lines = open("data/pairs.csv").readlines()
    for i in range(1, len(lines)):
        values.append(lines[i].strip().split(",")[column])
    return values

# Shrinks every pair and saves it as PNG.
def prepare_dataset():
    names = read_pairs_csv(0)
    for i in range(len(names)):
        folder = "data/raw/" + names[i]
        if os.path.exists(folder + "/before.jpg") == False or os.path.exists(folder + "/after.jpg") == False:
            print("Skipping " + names[i] + " (no before.jpg / after.jpg)")
            continue
        before = cv2.imread(folder + "/before.jpg")
        after = cv2.imread(folder + "/after.jpg")
        os.makedirs("data/pairs/" + names[i], exist_ok=True)
        cv2.imwrite("data/pairs/" + names[i] + "/before.png", shrink_to_width(before, 1100))
        cv2.imwrite("data/pairs/" + names[i] + "/after.png", shrink_to_width(after, 1100))
        print("Prepared " + names[i])
        
if __name__ == "__main__":
    prepare_dataset()