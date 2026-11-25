# CS 415 Final Project
# Before and After Scene Change Detection\

Aligns an AFTER photo onto a BEFORE photo and boxes what changed.
preprocess -> align -> change map -> segment -> boxes

## Files
    - src/preprocess.py: smoothing, shrinking photos
    - src/align.py: SIFT / Harris, RANSAC, warp
    - src/detect.py: gray / color / Sobel / Canny change maps
    - src/cnn_features.py: VGG16 change map
    - src/segment.py: threshold, opening/closing, regions
    - src/pipeline.py: runs all stages
    - src/evaluate.py: precision / recall / F1 / IoU
    - src/app.py: webcam demo
    - src/label_masks.py: ground-truth boxes
    - data/pairs.csv: list of pairs

## Setup
    - pip install -r requirements.txt
    - python src/cnn_features.py
    - Run everything from the top folder.

## Steps
    - Demo: python src/app.py (set camera_number at the top)
    - Results (once, after taking photos):
    - python src/preprocess.py -> shrinks photos
    - python src/label_masks.py -> answer key
    - python src/evaluate.py -> scores in results/
    - Optional: python src/pipeline.py -> stages for one pair (set pair_name)

## Demo keys
    |   key    |             action                 |
    |    b / a | capture before / after             |
    |    1 / 2 | SIFT / Harris                      |
    |    3-7   | gray / color / Sobel / Canny / CNN |
    |    n     | alignment on/off                   |
    |    o     | Otsu on/off                        |
    |    s     | save                               |
    |    q     | quit                               |