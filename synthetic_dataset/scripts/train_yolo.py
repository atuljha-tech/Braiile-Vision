"""
Train YOLOv8n on synthetic Braille dot boxes.
Run from repo root: python3 synthetic_dataset/scripts/train_yolo.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO_DIR = os.path.join(ROOT, "synthetic_dataset/yolo")
DATA = os.path.join(YOLO_DIR, "data.yaml")
OUT = os.path.join(ROOT, "synthetic_dataset/models/braille_dots_yolov8n.pt")


def main():
    if not os.path.isfile(DATA):
        raise SystemExit("Missing data.yaml — run generate_yolo_dataset.py first")

    from ultralytics import YOLO

    abs_yaml = os.path.join(YOLO_DIR, "data_train.yaml")
    with open(abs_yaml, "w") as f:
        f.write(
            f"path: {YOLO_DIR}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names: ['braille_dot']\n"
        )
    model = YOLO("yolov8n.pt")
    model.train(
        data=abs_yaml,
        epochs=12,
        imgsz=416,
        batch=8,
        project=os.path.join(ROOT, "synthetic_dataset/models/yolo_runs"),
        name="braille_dots",
        exist_ok=True,
        verbose=True,
    )
    best = os.path.join(
        ROOT, "synthetic_dataset/models/yolo_runs/braille_dots/weights/best.pt"
    )
    if os.path.isfile(best):
        import shutil
        shutil.copy(best, OUT)
        print(f"Saved {OUT}")
    else:
        print("Training finished; copy best.pt to", OUT)


if __name__ == "__main__":
    main()
