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
from pipeline import default_settings, load_pair, run_alignment, run_detection

# Precision, recall, F1, IoU and false-positive rate, counted inside the valid region.
# If nothing truly changed in the pair, only the false-positive rate is returned (others None).
def mask_metrics(pred, gt, valid):
    pred = pred & valid
    gt = gt & valid
    tp = np.sum(pred & gt)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    unchanged = np.sum(valid) - np.sum(gt)
    fp_rate = fp / unchanged

    if tp + fn == 0:
        return None, None, None, None, fp_rate
    precision = 0.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    iou = tp / (tp + fp + fn)
    return precision, recall, f1, iou, fp_rate

# Runs the pipeline on every pair with a gt.png and returns one row (dictionary) per pair.
# cache saves alignment results, so experiments that only change stages C-D run faster.
def evaluate_dataset(s, cache=None):
    names = read_pairs_csv(0)
    conditions = read_pairs_csv(2)
    rows = []
    for i in range(len(names)):
        name = names[i]
        if os.path.exists("data/pairs/" + name + "/gt.png") == False:
            continue
        gt = cv2.imread("data/pairs/" + name + "/gt.png", 0) > 127

        key = name + "_" + s["feature_method"] + "_" + str(s["use_alignment"]) + "_" + str(s["equalize"])
        if cache is not None and key in cache:
            data = cache[key]
        else:
            before, after = load_pair(name)
            data = run_alignment(before, after, s)
            if cache is not None:
                cache[key] = data
        data = run_detection(data, s)

        precision, recall, f1, iou, fp_rate = mask_metrics(data["mask"], gt, data["valid"])
        row = {"pair": name, "condition": conditions[i],
               "precision": precision, "recall": recall, "f1": f1, "iou": iou, "fp_rate": fp_rate,
               "align_ok": data["align_ok"], "matches": len(data["matches"]),
               "inliers": int(np.sum(data["inliers"])), "threshold": data["threshold"],
               "regions": len(data["boxes"]), "time": data["total_time"]}
        rows.append(row)
        print(name, "F1:", show(f1), "IoU:", show(iou), "time:", show(data["total_time"]))
    return rows

# Rounds a number for printing
def show(value):
    if value is None:
        return "-"
    return str(round(float(value), 2))

# Average of one column, ignoring blanks
def average(rows, column):
    total = 0
    count = 0
    for i in range(len(rows)):
        if rows[i][column] is not None:
            total += rows[i][column]
            count += 1
    if count == 0:
        return None
    return total / count

# Writing to csv
def save_rows(rows, path):
    columns = list(rows[0].keys())
    f = open(path, "w")
    f.write(",".join(columns) + "\n")
    for i in range(len(rows)):
        values = []
        for column in columns:
            values.append(str(rows[i][column]))
        f.write(",".join(values) + "\n")
    f.close()

if __name__ == "__main__":
    rows = evaluate_dataset(default_settings())
    print("Average F1:", show(average(rows, "f1")), "IoU:", show(average(rows, "iou")))
    os.makedirs("results", exist_ok=True)
    save_rows(rows, "results/default_metrics.csv")