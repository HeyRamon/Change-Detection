"""
Ramon Vazquez, Nathan Pebler
Professor Tang
CS 415 Final Project
11/24/26
"""
import os

import cv2
import numpy as np

from preprocess import shrink_to_width
from pipeline import default_settings, load_pair, run_alignment, run_detection, make_panel, thresholds, put_label

# the webcam's number (change this number if it is different)
camera_number = 0
pair_name = ""

def nothing(value):
    pass

# Waits 30 ms for a key and returns it as a letter
def get_key():
    key = cv2.waitKey(30)
    if key == -1:
        return ""
    return chr(key % 256)

# Reads a few frames to avoid camera buffer issues
def capture(camera):
    for i in range(5):
        ok, frame = camera.read()
    frame = shrink_to_width(frame, 800)
    return frame

# Shows all stages plus one line with the current settings
def show(data, settings):
    panel = np.zeros((630, 1600, 3), dtype=np.uint8)
    panel[0:600, 0:1600] = make_panel(data)

    text = "features: " + settings["feature_method"] + " | change map: " + settings["detect_method"]
    text = text + " | threshold: " + str(round(data["threshold"], 2))
    if settings["use_otsu"]:
        text = text + " (otsu)"
    text = text + " | align: " + str(settings["use_alignment"])
    text = text + " | time: " + str(round(data["total_time"], 2)) + " s"
    if np.sum(data["mask"]) > 0.5 * np.sum(data["valid"]):
        text = text + " | GLOBAL CHANGE (lighting?)"
    cv2.putText(panel, text, (5, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imshow("Change Detection", panel)
    return panel

# Saves the panel and the before/after images
def save(panel, before, after):
    number = 1
    while os.path.exists("results/demo_" + str(number) + ".png"):
        number += 1
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/captured/live_" + str(number), exist_ok=True)
    cv2.imwrite("results/demo_" + str(number) + ".png", panel)
    cv2.imwrite("data/captured/live_" + str(number) + "/before.png", before)
    cv2.imwrite("data/captured/live_" + str(number) + "/after.png", after)
    print("Saved demo_" + str(number) + ".png")

def main():
    settings = default_settings()
    before = None
    after = None
    aligned = None
    panel = None
    rerun_align = False
    rerun_detect = False
    if pair_name != "":
        before, after = load_pair(pair_name)
        rerun_align = True

    # WINDOW_NORMAL lets you resize the window, trackbar being the threshold slider
    cv2.namedWindow("Change Detection", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("threshold x100", "Change Detection", int(thresholds["color"] * 100), 100, nothing)
    cv2.imshow("Change Detection", np.zeros((300, 800, 3), dtype=np.uint8))

    camera = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
    if camera.isOpened() == False:
        print("No webcam found")
        camera = None

    while True:
        if rerun_align:
            print("Aligning...")
            aligned = run_alignment(before, after, settings)
            rerun_align = False
            rerun_detect = True
        if rerun_detect:
            settings["threshold"] = cv2.getTrackbarPos("threshold x100", "Change Detection") / 100
            data = run_detection(aligned, settings)
            panel = show(data, settings)
            rerun_detect = False

        if camera is not None:
            ok, frame = camera.read()
            if ok:
                text = "b = before, a = after, q = quit"
                put_label(frame, text, 0, 0, len(text) * 10 + 15, 0.8)
                cv2.imshow("Camera", frame)

        key = get_key()

        if aligned is not None:
            slider = cv2.getTrackbarPos("threshold x100", "Change Detection") / 100
            if slider != settings["threshold"]:
                rerun_detect = True

        if before is not None and after is not None:
            have_pair = True
        else:
            have_pair = False

        if key == "q":
            break
        elif key == "b":
            if camera is not None:
                before = capture(camera)
                print("Captured BEFORE")
        elif key == "a":
            if camera is not None and before is not None:
                after = capture(camera)
                print("Captured AFTER")
                rerun_align = True
        elif key == "1":
            settings["feature_method"] = "sift"
            rerun_align = have_pair
        elif key == "2":
            settings["feature_method"] = "harris"
            rerun_align = have_pair
        elif key == "3" or key == "4" or key == "5" or key == "6" or key == "7":
            if key == "3":
                settings["detect_method"] = "gray"
            elif key == "4":
                settings["detect_method"] = "color"
            elif key == "5":
                settings["detect_method"] = "sobel"
            elif key == "6":
                settings["detect_method"] = "canny"
            else:
                settings["detect_method"] = "cnn"
            cv2.setTrackbarPos("threshold x100", "Change Detection", int(thresholds[settings["detect_method"]] * 100))
            if aligned is not None:
                rerun_detect = True
        elif key == "n":
            if settings["use_alignment"]:
                settings["use_alignment"] = False
            else:
                settings["use_alignment"] = True
            rerun_align = have_pair
        elif key == "o":
            if settings["use_otsu"]:
                settings["use_otsu"] = False
            else:
                settings["use_otsu"] = True
            if aligned is not None:
                rerun_detect = True
        elif key == "s":
            if panel is not None:
                save(panel, before, after)

    if camera is not None:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()