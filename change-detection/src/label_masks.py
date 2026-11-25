"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import os

import cv2
import numpy as np

from preprocess import read_pairs_csv
from pipeline import default_settings, load_pair, run_alignment

# Fills each box with white in the mask.
def add_boxes(mask, boxes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        mask[y:y + h, x:x + w] = 255


if __name__ == "__main__":
    names = read_pairs_csv(0)
    for i in range(len(names)):
        folder = "data/pairs/" + names[i]
        if os.path.exists(folder + "/before.png") == False or os.path.exists(folder + "/gt.png"):
            continue
        print("Labeling " + names[i])
        before, after = load_pair(names[i])
        warped = run_alignment(before, after, default_settings())["warped_color"]

        mask = np.zeros((before.shape[0], before.shape[1]), dtype=np.uint8)
        boxes = cv2.selectROIs("BEFORE: objects that disappeared or moved away (Esc = done)", before)
        add_boxes(mask, boxes)
        boxes = cv2.selectROIs("AFTER: objects that appeared or moved here (Esc = done)", warped)
        add_boxes(mask, boxes)
        cv2.destroyAllWindows()

        cv2.imwrite(folder + "/gt.png", mask)
        print("Saved " + folder + "/gt.png")