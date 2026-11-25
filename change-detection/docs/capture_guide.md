## Rules
    - Keep something textured in view (books, papers, keyboard).
    - Changes should be clear objects, about mug size or bigger.
    - No people in the frame.
    - Camera types:
      - still: phone resting on books, don't touch it
      - shift: move the phone a few cm between shots
      - rotation: turn the phone 10-15 degrees
    - Take each pair twice and keep the better one.

## Saving the photos
    - iPhone: Settings > Camera > Formats > Most Compatible (JPG, not HEIC).
    - Put them in data/raw/<name>/before.jpg and after.jpg (name from the table).
    - Run python src/preprocess.py.
    - Delete lines in data/pairs.csv for pairs you skip.

## How to Run
    - Run python src/label_masks.py.
    - BEFORE window: box what disappeared or moved away. Enter after each box, Esc when done.
    - AFTER window: box what appeared or moved here. Enter after each box, Esc when done.
    - Don't box shadows or lighting changes.
    - Nothing changed: press Esc twice.