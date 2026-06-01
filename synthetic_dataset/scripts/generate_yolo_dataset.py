"""
Generate YOLO-format dataset: composite Braille pages with dot bounding boxes.
Run from repo root: python3 synthetic_dataset/scripts/generate_yolo_dataset.py
"""
import os
import random
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFilter

# Reuse Braille layout from cell generator
BRAILLE_MAP = {
    "A": [1], "B": [1, 2], "C": [1, 4], "D": [1, 4, 5], "E": [1, 5],
    "F": [1, 2, 4], "G": [1, 2, 4, 5], "H": [1, 2, 5], "I": [2, 4], "J": [2, 4, 5],
    "K": [1, 3], "L": [1, 2, 3], "M": [1, 3, 4], "N": [1, 3, 4, 5], "O": [1, 3, 5],
    "P": [1, 2, 3, 4], "Q": [1, 2, 3, 4, 5], "R": [1, 2, 3, 5], "S": [2, 3, 4],
    "T": [2, 3, 4, 5], "U": [1, 3, 6], "V": [1, 2, 3, 6], "W": [2, 4, 5, 6],
    "X": [1, 3, 4, 6], "Y": [1, 3, 4, 5, 6], "Z": [1, 3, 5, 6],
}
DOT_POS = {1: (20, 20), 2: (20, 50), 3: (20, 80), 4: (60, 20), 5: (60, 50), 6: (60, 80)}

ROOT = "synthetic_dataset/yolo"
IMG_W, IMG_H = 640, 480
CELL_W, CELL_H = 100, 120
N_TRAIN, N_VAL = 320, 80


def _draw_cell(char: str, jitter: bool = True) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """Return cell image and list of dot bboxes (x1,y1,x2,y2) in cell coords."""
    img = Image.new("L", (CELL_W, CELL_H), 255)
    draw = ImageDraw.Draw(img)
    boxes = []
    for dot in BRAILLE_MAP[char]:
        x, y = DOT_POS[dot]
        if jitter:
            x += random.randint(-4, 4)
            y += random.randint(-4, 4)
        r = random.randint(8, 13)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
        boxes.append((x - r, y - r, x + r, y + r))
    angle = random.uniform(-6, 6)
    img = img.rotate(angle, expand=False, fillcolor=255)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return img, boxes


def _make_page(split: str, idx: int) -> None:
    canvas = Image.new("L", (IMG_W, IMG_H), 255)
    labels = []
    n_cells = random.randint(3, 10)
    letters = random.choices(list(BRAILLE_MAP.keys()), k=n_cells)

    for _ in letters:
        cell, boxes = _draw_cell(random.choice(letters))
        ox = random.randint(10, max(10, IMG_W - CELL_W - 10))
        oy = random.randint(10, max(10, IMG_H - CELL_H - 10))
        canvas.paste(cell, (ox, oy))
        for x1, y1, x2, y2 in boxes:
            gx1, gy1 = ox + x1, oy + y1
            gx2, gy2 = ox + x2, oy + y2
            cx = ((gx1 + gx2) / 2) / IMG_W
            cy = ((gy1 + gy2) / 2) / IMG_H
            bw = (gx2 - gx1) / IMG_W
            bh = (gy2 - gy1) / IMG_H
            labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    img_dir = os.path.join(ROOT, "images", split)
    lbl_dir = os.path.join(ROOT, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    name = f"page_{idx:04d}"
    canvas.save(os.path.join(img_dir, f"{name}.jpg"), quality=92)
    with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(labels))


def main():
    for split, n in [("train", N_TRAIN), ("val", N_VAL)]:
        for i in range(n):
            _make_page(split, i)
    print(f"YOLO dataset ready: {N_TRAIN} train + {N_VAL} val under {ROOT}/")


if __name__ == "__main__":
    main()
