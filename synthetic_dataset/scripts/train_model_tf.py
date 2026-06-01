"""
Train TensorFlow/Keras CNN on the same synthetic cell dataset as PyTorch.
Run from repo root: python3 synthetic_dataset/scripts/train_model_tf.py
"""
import os
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_PATH = "synthetic_dataset/generated"
MODEL_SAVE_PATH = "synthetic_dataset/models/braille_cnn_tf.keras"
IMAGE_SIZE = 64
EPOCHS = 3
BATCH_SIZE = 16


def load_dataset():
    X, y = [], []
    letters = sorted(os.listdir(DATASET_PATH))
    letter_to_idx = {c: i for i, c in enumerate(letters)}
    for char in letters:
        folder = os.path.join(DATASET_PATH, char)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".png", ".jpg")):
                continue
            path = os.path.join(folder, fname)
            img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            X.append(arr[..., np.newaxis])
            y.append(letter_to_idx[char])
    return np.array(X), np.array(y), letters


def build_model(num_classes: int):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def main():
    import tensorflow as tf

    X, y, letters = load_dataset()
    print(f"Loaded {len(X)} images, {len(letters)} classes")

    model = build_model(len(letters))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    with open("synthetic_dataset/models/tf_class_labels.txt", "w") as f:
        f.write("\n".join(letters))
    print(f"Saved {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
