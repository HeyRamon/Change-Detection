"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import cv2
import numpy as np

from preprocess import gaussian_filter, compute_gradient

def non_maxima_suppression(cornerness, threshold, kernel_size=5):
    height, width = cornerness.shape
    res = np.zeros_like(cornerness)
    kernel_radius = int(kernel_size/2)
    neighbors_mask = np.ones((kernel_size, kernel_size), dtype='bool')
    neighbors_mask[kernel_radius, kernel_radius] = False
    for y in range(kernel_radius, height-kernel_radius):
        for x in range(kernel_radius, width-kernel_radius):
            if cornerness[y, x] < threshold:
                continue
            left = x - kernel_radius
            top = y - kernel_radius
            patch = cornerness[top:top+kernel_size, left:left+kernel_size]
            neighbors = patch[neighbors_mask]
            if cornerness[y, x] >= neighbors.max():
                res[y, x] = 1
    return res

# Harris corners
def harris_corners(im, max_corners=1500):
    im = im / 255

    Ix, Iy = compute_gradient(im)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    sigma = 1.5
    Hx2 = gaussian_filter(Ix2, sigma)
    Hy2 = gaussian_filter(Iy2, sigma)
    Hxy = gaussian_filter(Ixy, sigma)

    cornerness = (Hx2*Hy2 - Hxy*Hxy)/(Hx2 + Hy2 + 10**-8)

    threshold = 0.01 * cornerness.max()

    cornerness_nms = non_maxima_suppression(cornerness, threshold, 7)

    y_coordinates, x_coordinates = np.where(cornerness_nms > 0)

    strengths = cornerness[y_coordinates, x_coordinates]
    if len(strengths) > max_corners:
        sorted_indices = np.argsort(strengths)
        cutoff = strengths[sorted_indices[len(strengths) - max_corners]]
        x_coordinates = x_coordinates[strengths >= cutoff]
        y_coordinates = y_coordinates[strengths >= cutoff]
    return x_coordinates, y_coordinates

# 8x8 patch around each corner, normalized.
def patch_descriptors(im, x_coordinates, y_coordinates):
    im_blurred = gaussian_filter(im, 2.5)
    height, width = im.shape
    keypoints = []
    descriptors = []
    for i in range(len(x_coordinates)):
        x = x_coordinates[i]
        y = y_coordinates[i]
        if x < 20 or y < 20 or x >= width - 20 or y >= height - 20:
            continue
        patch = []
        for dy in range(-20, 20, 5):
            for dx in range(-20, 20, 5):
                patch.append(im_blurred[y + dy, x + dx])
        patch = np.array(patch)
        patch = (patch - patch.mean()) / (patch.std() + 10**-8)
        keypoints.append(cv2.KeyPoint(float(x), float(y), 40.0))
        descriptors.append(patch)
    return keypoints, np.array(descriptors)

# Keypoints and descriptors
def detect_and_describe(im, method):
    if method == "sift":
        sift = cv2.SIFT_create(nfeatures=1500)
        keypoints, descriptors = sift.detectAndCompute(im.astype(np.uint8), None)
        return keypoints, descriptors
    x_coordinates, y_coordinates = harris_corners(im)
    return patch_descriptors(im, x_coordinates, y_coordinates)

def match_features(descriptors1, descriptors2):
    good_matches = []
    ratio_thresh = 0.75

    for i, desc1 in enumerate(descriptors1):
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        sorted_indices = np.argsort(distances)

        if len(sorted_indices) > 1:
            best_match = sorted_indices[0]
            second_best_match = sorted_indices[1]

            # Apply ratio test
            if distances[best_match] < ratio_thresh * distances[second_best_match]:
                good_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=int(best_match),
                                               _distance=float(distances[best_match])))

    return good_matches

def compute_homography(points1, points2):
    A = []
    for i in range(len(points1)):
        x, y = points1[i]
        xp, yp = points2[i]
        A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])
    U, S, Vt = np.linalg.svd(np.array(A))
    h = Vt[8] / Vt[8][8]
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], h[8]]])
    return H

def apply_homography(H, x, y):
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    new_x = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    new_y = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    return new_x, new_y

def ransac_homography(points1, points2, num_iterations=1000, threshold=3.0):
    np.random.seed(0)
    best_inliers = np.zeros(len(points1), dtype=bool)
    if len(points1) < 4:
        return None, best_inliers

    for i in range(num_iterations):
        sample = np.random.choice(len(points1), 4, replace=False)
        H = compute_homography(points1[sample], points2[sample])
        new_x, new_y = apply_homography(H, points1[:, 0], points1[:, 1])
        errors = np.sqrt((new_x - points2[:, 0]) ** 2 + (new_y - points2[:, 1]) ** 2)
        inliers = errors < threshold
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers

    if np.sum(best_inliers) < 4:
        return None, best_inliers
    H = compute_homography(points1[best_inliers], points2[best_inliers])
    return H, best_inliers

def identity_matrix():
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])

# Matches before and after and estimates H that maps before points to after points.
def align_images(before_gray, after_gray, method="sift", min_inliers=15):
    keypoints1, descriptors1 = detect_and_describe(before_gray, method)
    keypoints2, descriptors2 = detect_and_describe(after_gray, method)
    if len(keypoints1) < 4 or len(keypoints2) < 4:
        return identity_matrix(), False, keypoints1, keypoints2, [], np.zeros(0, dtype=bool)

    matches = match_features(descriptors1, descriptors2)

    points1 = np.zeros((len(matches), 2))
    points2 = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        points1[i] = keypoints1[matches[i].queryIdx].pt
        points2[i] = keypoints2[matches[i].trainIdx].pt

    H, inliers = ransac_homography(points1, points2)

    ok = True
    if H is None or np.sum(inliers) < min_inliers:
        ok = False
        H = identity_matrix()
    return H, ok, keypoints1, keypoints2, matches, inliers

# Inverse warping with bilinear interpolation, one row at a time
# For each output pixel we look up its position in the after image
def warp(im, H, height, width):
    im = im.astype(float)
    in_height, in_width = im.shape
    out = np.zeros((height, width))
    valid = np.zeros((height, width), dtype=bool)
    x_out = np.arange(width)
    for y_out in range(height):
        x, y = apply_homography(H, x_out, y_out)
        valid[y_out] = (x >= 0) & (y >= 0) & (x <= in_width - 1) & (y <= in_height - 1)

        x = np.clip(x, 0, in_width - 1)
        y = np.clip(y, 0, in_height - 1)
        x0 = x.astype(int)
        y0 = y.astype(int)
        x1 = np.clip(x0 + 1, 0, in_width - 1)
        y1 = np.clip(y0 + 1, 0, in_height - 1)
        fx = x - x0
        fy = y - y0
        out[y_out] = ((1 - fx) * (1 - fy) * im[y0, x0] + fx * (1 - fy) * im[y0, x1] + (1 - fx) * fy * im[y1, x0] + fx * fy * im[y1, x1])
    return out, valid