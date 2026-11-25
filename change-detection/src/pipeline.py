"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess
from align import align_images, warp, identity_matrix
from detect import compute_change_map
from segment import segment_changes, erode

thresholds = {"gray": 0.15, "color": 0.20, "sobel": 0.30, "canny": 0.40}

def default_settings():
    return {"sigma": 1.0,
            "use_alignment": True,
            "feature_method": "sift",
            "detect_method": "color",
            "threshold": None,
            "use_otsu": False,
            "open_size": 5,
            "close_size": 15,
            "min_area": 400}

def load_pair(name):
    before = cv2.imread("data/pairs/" + name + "/before.png")
    after = cv2.imread("data/pairs/" + name + "/after.png")
    return before, after

# seconds since start
def seconds_since(start):
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()

# preprocess, align and warp the after image onto the before image.
def run_alignment(before, after, settings):
    start = cv2.getTickCount()
    before_color, before_gray = preprocess(before, settings["sigma"])
    after_color, after_gray = preprocess(after, settings["sigma"])

    if settings["use_alignment"]:
        H, ok, keypoints1, keypoints2, matches, inliers = align_images(before_gray, after_gray, settings["feature_method"])
    else:
        H = identity_matrix()
        ok = True
        keypoints1 = []
        keypoints2 = []
        matches = []
        inliers = np.zeros(0, dtype='bool')
    align_time = seconds_since(start)

    start = cv2.getTickCount()
    height, width = before_gray.shape
    warped_gray, valid = warp(after_gray, H, height, width)
    warped_color = np.zeros(before_color.shape)
    for c in range(3):
        warped_color[:, :, c], valid = warp(after_color[:, :, c], H, height, width)
    valid = erode(valid, 21)

    # outside the valid area, copy the before pixels so the border is never a change
    warped_gray[valid == False] = before_gray[valid == False]
    warped_color[valid == False] = before_color[valid == False]
    warped_color = np.clip(warped_color, 0, 255).astype(np.uint8)
    warp_time = seconds_since(start)

    return {"before": before, "after": after,
            "before_color": before_color, "before_gray": before_gray, "after_gray": after_gray,
            "warped_color": warped_color, "warped_gray": warped_gray, "valid": valid,
            "H": H, "align_ok": ok, "keypoints1": keypoints1, "keypoints2": keypoints2,
            "matches": matches, "inliers": inliers,
            "align_time": align_time, "warp_time": warp_time}

# change map and segmentation
def run_detection(data, settings):
    start = cv2.getTickCount()
    data["change"] = compute_change_map(settings["detect_method"], data["before_color"], data["before_gray"],
                                        data["warped_color"], data["warped_gray"], data["valid"])
    data["detect_time"] = seconds_since(start)

    if settings["threshold"] is None:
        threshold = thresholds[settings["detect_method"]]
    else:
        threshold = settings["threshold"]

    start = cv2.getTickCount()
    mask, boxes, threshold = segment_changes(data["change"], data["valid"], threshold, settings["use_otsu"],
                                             settings["open_size"], settings["close_size"], settings["min_area"])
    data["mask"] = mask
    data["boxes"] = boxes
    data["threshold"] = threshold
    data["segment_time"] = seconds_since(start)
    data["total_time"] = data["align_time"] + data["warp_time"] + data["detect_time"] + data["segment_time"]
    data["settings"] = settings
    return data

def run_pipeline(before, after, settings):
    data = run_alignment(before, after, settings)
    return run_detection(data, settings)

def put_label(im, text, x, y, width, scale=0.5):
    height = int(40 * scale)
    im[y:y + height, x:x + width] = 0
    cv2.putText(im, text, (x + 5, y + int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)

# RANSAC inlier matches as white lines
def draw_matches(data):
    inlier_matches = []
    for i in range(len(data["matches"])):
        if data["inliers"][i]:
            inlier_matches.append(data["matches"][i])

    step = len(inlier_matches) // 50 + 1
    shown_matches = []
    for i in range(0, len(inlier_matches), step):
        shown_matches.append(inlier_matches[i])

    return cv2.drawMatches(data["before"], data["keypoints1"], data["after"], data["keypoints2"], shown_matches, None,
                           matchColor=(255, 255, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Red = before, cyan = aligned after, grey is in the middle.
def draw_overlay(data):
    out = np.zeros(data["before_color"].shape)
    out[:, :, 2] = data["before_gray"]
    out[:, :, 1] = data["warped_gray"]
    out[:, :, 0] = data["warped_gray"]
    return out.astype(np.uint8)

# Aligned after image with a red box around each changed region
def draw_result(data):
    out = data["warped_color"].copy()
    for i in range(len(data["boxes"])):
        x, y, w, h = data["boxes"][i]
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        cv2.line(out, (x1, y1), (x2, y1), (0, 0, 255), 2)
        cv2.line(out, (x2, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(out, (x2, y2), (x1, y2), (0, 0, 255), 2)
        cv2.line(out, (x1, y2), (x1, y1), (0, 0, 255), 2)
    return out

# titles
def stage_images(data):
    settings = data["settings"]
    num_inliers = int(np.sum(data["inliers"]))
    if data["align_ok"]:
        match_title = settings["feature_method"] + ": " + str(len(data["matches"])) + " matches, " + str(num_inliers) + " inliers"
    else:
        match_title = "ALIGNMENT FAILED"
    change = (data["change"] * 255).astype(np.uint8)
    mask = data["mask"].astype(np.uint8) * 255
    return [["Before", data["before"]],
            ["After", data["after"]],
            [match_title, draw_matches(data)],
            ["Overlay (red = before, cyan = after)", draw_overlay(data)],
            ["Change map: " + settings["detect_method"], cv2.cvtColor(change, cv2.COLOR_GRAY2BGR)],
            ["Mask (threshold " + str(round(data["threshold"], 2)) + ")", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)],
            ["Result: " + str(len(data["boxes"])) + " regions", draw_result(data)]]

def make_panel(data, tile_width=400, tile_height=300):
    panel = np.zeros((2 * tile_height, 4 * tile_width, 3), dtype=np.uint8)
    images = stage_images(data)
    rows = [0, 0, 0, 1, 1, 1, 1]
    cols = [0, 1, 2, 0, 1, 2, 3]
    widths = [1, 1, 2, 1, 1, 1, 1]
    for i in range(len(images)):
        title = images[i][0]
        im = images[i][1]
        height, width, channels = im.shape
        box_width = widths[i] * tile_width

        # keep every step-th pixel so the image fits under its 20 px title bar
        step = 1
        while height / step > tile_height - 20 or width / step > box_width:
            step += 1
        small = im[::step, ::step]

        top = rows[i] * tile_height
        left = cols[i] * tile_width
        panel[top + 20:top + 20 + small.shape[0], left:left + small.shape[1]] = small
        put_label(panel, title, left, top, box_width)
    return panel

# Saves in matplotlib
def show_figure(data):
    panel = make_panel(data)
    plt.imshow(panel[:, :, ::-1])  # BGR -> RGB
    plt.show()

if __name__ == "__main__":
    name = "desk01"
    before, after = load_pair(name)
    data = run_pipeline(before, after, default_settings())

    print(name, "-", len(data["matches"]), "matches,", int(np.sum(data["inliers"])), "inliers,",
          len(data["boxes"]), "regions")
    print("time (s): align", round(data["align_time"], 2), "warp", round(data["warp_time"], 2),
          "detect", round(data["detect_time"], 2), "segment", round(data["segment_time"], 2),
          "total", round(data["total_time"], 2))

    show_figure(data)