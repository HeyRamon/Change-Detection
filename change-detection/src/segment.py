"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import cv2
import numpy as np


# Otsu threshold: the t that maximizes the between-class variance.
def otsu_threshold(change, valid):
    levels = (change[valid] * 255).astype(int)
    hist = np.zeros(256)
    for value in range(256):
        hist[value] = np.sum(levels == value)
    hist /= hist.sum()
    values = np.arange(256)

    best_t = 128
    best_variance = 0
    for t in range(1, 256):
        w0 = hist[:t].sum()
        w1 = hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mean0 = (hist[:t] * values[:t]).sum() / w0
        mean1 = (hist[t:] * values[t:]).sum() / w1
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > best_variance:
            best_variance = variance
            best_t = t
    return best_t / 255


# Number of True pixels in each size x size neighborhood.
def count_neighbors(mask, size):
    kernel = np.ones((size, size))
    counts = cv2.filter2D(mask.astype(float), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return counts


# Erosion: a pixel stays True only if its whole neighborhood is True.
def erode(mask, size):
    counts = count_neighbors(mask, size)
    return counts > size * size - 0.5


# Dilation: a pixel becomes True if any pixel in its neighborhood is True.
def dilate(mask, size):
    counts = count_neighbors(mask, size)
    return counts > 0.5


# Opening removes small noise; closing fills small holes.
def opening(mask, size):
    mask = erode(mask, size)
    mask = dilate(mask, size)
    return mask


def closing(mask, size):
    mask = dilate(mask, size)
    mask = erode(mask, size)
    return mask


# Labels 8-connected regions with a stack flood fill (like edge_linking in MP2).
# Returns the label image and a list of (area, x, y, w, h) per region.
def connected_components(mask):
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=int)
    regions = []
    for start_row, start_col in np.argwhere(mask):
        if labels[start_row, start_col] != 0:
            continue
        label = len(regions) + 1
        labels[start_row, start_col] = label
        stack = [(start_row, start_col)]
        rows = []
        cols = []
        while stack:
            row, col = stack.pop()
            rows.append(row)
            cols.append(col)
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    next_row, next_col = row + d_row, col + d_col
                    inside = 0 <= next_row < height and 0 <= next_col < width
                    if inside and mask[next_row, next_col] and labels[next_row, next_col] == 0:
                        labels[next_row, next_col] = label
                        stack.append((next_row, next_col))
        area = len(rows)
        x = min(cols)
        y = min(rows)
        w = max(cols) - x + 1
        h = max(rows) - y + 1
        regions.append((area, x, y, w, h))
    return labels, regions


# Threshold -> opening -> closing -> drop small regions. Returns mask, boxes, threshold.
def segment_changes(change, valid, threshold, use_otsu, open_size, close_size, min_area):
    if use_otsu:
        threshold = otsu_threshold(change, valid)

    mask = np.zeros(change.shape, dtype='bool')
    mask[change > threshold] = True
    mask[valid == False] = False
    mask = opening(mask, open_size)
    mask = closing(mask, close_size)
    mask[valid == False] = False

    # more than half the image changed: almost always lighting, so no boxes
    if np.sum(mask) > 0.5 * np.sum(valid):
        return mask, [], threshold

    labels, regions = connected_components(mask)
    final_mask = np.zeros(mask.shape, dtype='bool')
    boxes = []
    for i in range(len(regions)):
        area, x, y, w, h = regions[i]
        if area >= min_area:
            final_mask[labels == i + 1] = True
            boxes.append((x, y, w, h))
    return final_mask, boxes, threshold