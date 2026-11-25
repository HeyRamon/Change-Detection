# Dataset capture guide and shot list

Goal: about 17 core pairs + 4 failure-case pairs, captured together in about 1.5 hours.
Each pair = one `before` photo and one `after` photo of the same scene.

## Capture rules (read before shooting)

1. **Same camera, same zoom, landscape orientation** for both photos of a pair. Use one phone for everything.
2. **Lock exposure and focus.** Auto-exposure changes brightness between shots and creates fake "lighting" changes.
   - iPhone: long-press on the scene until "AE/AF LOCK" appears.
   - Android: tap-and-hold to lock, or use Pro/Manual mode.
   - Turn **off** HDR, "scene optimizer", beauty and night modes.
3. **Texture matters for alignment.** Make sure the frame has books, posters, a keyboard, or printed paper.
   A completely blank desk or wall is a *failure case* (fail02), not a normal pair.
4. **Camera motion levels** (written in the `condition` column):
   - *still*: phone resting on a stack of books or a mini tripod; do not touch it between shots (use the timer or a remote).
   - *shift*: move the phone 2-5 cm or tilt it a few degrees between shots (handheld is fine).
   - *rotation*: rotate the phone about 10-15 degrees around the lens axis.
5. **Changes should be clear objects**, at least about the size of a mug in the frame. Use objects whose color differs from the background in some pairs and is similar to it in others.
6. **No people or moving things** in the frame (except the outdoor pair where a parked car or bike changes).
7. Shoot each pair **twice** if possible and keep the better one. This takes seconds and saves a reshoot.

## Folder naming

Copy the photos to `data/raw/<name>/before.jpg` and `data/raw/<name>/after.jpg`, with `<name>` from the table below
(the file names must be exactly `before.jpg` and `after.jpg`).
**iPhone:** photos are often HEIC, which OpenCV cannot read. Set Settings > Camera > Formats > "Most Compatible" before
shooting, or export as JPG.
Then run `python src/preprocess.py` (it shrinks them to about 1000 px wide and saves PNGs in `data/pairs`).
Delete the lines in `data/pairs.csv` for pairs you skip, and add lines for any new pairs.

## Shot list

| name | scene | what changes | camera | tests |
|---|---|---|---|---|
| desk01 | desk | add a mug | still | baseline |
| desk02 | desk | remove a book | still | removed objects |
| desk03 | desk | move the phone/stapler ~15 cm | still | moved = two regions |
| desk04 | desk | add an object | shift | alignment needed (exp. 1) |
| desk05 | desk | add 1, remove 1, move 1 object | shift | multiple regions |
| desk06 | desk | desk lamp on/off **and** add one object | still | lighting (exp. 3) |
| room01 | room | move a chair | shift | larger scene |
| room02 | room | add a backpack on the floor | shift | |
| room03 | room | lights on vs. daylight only **and** move one object | shift | lighting (exp. 3) |
| room04 | room | **nothing** changes | shift | false positives only |
| shelf01 | shelf | remove a book | shift | |
| shelf02 | shelf | swap two same-size objects of different color | shift | color vs. edges |
| shelf03 | shelf | add a small object (coin, eraser) | shift | min-area filter (exp. 4) |
| shelf04 | shelf | add an object | rotation 10-15 deg | SIFT vs. Harris+patch (exp. 2) |
| outdoor01 | outdoor | parked bike/car/trash bin moved | shift | outdoor |
| outdoor02 | outdoor | add a backpack on a bench or wall | shift | |
| outdoor03 | outdoor | **nothing** changes; shoot ~20 min apart (sun/clouds) | shift | lighting false positives |

### Failure cases (we expect these to fail; they go in the analysis)

| name | what to do | expected failure and why (course concept) |
|---|---|---|
| fail01 | desk with objects at different depths; walk ~50 cm sideways | **Parallax**: a homography is only valid for planar / far scenes or pure rotation (L18 slides 53-54) |
| fail02 | blank white desk or wall, add an object | **Low texture**: few corners / keypoints, so alignment is weak (L12) |
| fail03 | no object change; cast a strong shadow with a lamp or your hand (out of frame) | **Shadows**: shadow borders are illumination edges (L5 slide 4), so even edge methods flag them |
| fail04 | step back or zoom ~1.5x, and add an object | **Scale**: the Harris patch descriptor has no scale invariance, SIFT does (L14, L15) |

Optional: a pair with a glossy surface (monitor, window) for reflections.

## Ground-truth masks (about 1-2 minutes per pair)

Run `python src/label_masks.py`. For each pair, two windows open one after the other:

1. **BEFORE**: drag a box around each object that **disappeared or moved away**. Press Enter after each box, Esc when done.
2. **AFTER (aligned)**: drag a box around each object that **appeared or moved here**. Press Enter after each box, Esc when done.

The boxes are saved as `gt.png` (white = changed). Pairs that already have a `gt.png` are skipped. Delete the file to relabel a pair.

Rules (same for every pair):
- Box the object tightly.
- Do **not** box shadows, reflections or brightness changes. These are what the failure analysis measures.
- For "nothing changes" pairs, just press Esc twice. This saves an empty mask.

Notes for the report:
- Boxes include a little background around round objects, so precision and IoU are a bit lower than with exact outlines. This is the same for every method, so comparisons stay fair.
- The AFTER window uses our SIFT alignment. If alignment fails for a pair, its boxes may be shifted.

## Optional public dataset

**VL-CMU-CD** (Alcantarilla et al., street-view change detection) has 1,362 coarsely aligned street-scene pairs with
change masks. It is widely used, but the official download link has moved over the years. If you can find a working
copy quickly, running our pipeline on about 20 test pairs makes a nice extra table. Do not depend on it, and credit it if used.
