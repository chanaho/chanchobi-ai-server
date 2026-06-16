import os
import random
import shutil

IMAGE_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

OUT_IMAGE_TRAIN = "dataset/images/train"
OUT_IMAGE_VAL = "dataset/images/val"

OUT_LABEL_TRAIN = "dataset/labels/train"
OUT_LABEL_VAL = "dataset/labels/val"

TRAIN_RATIO = 0.8


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_dataset():
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    ensure_dir(OUT_IMAGE_TRAIN)
    ensure_dir(OUT_IMAGE_VAL)
    ensure_dir(OUT_LABEL_TRAIN)
    ensure_dir(OUT_LABEL_VAL)

    # TRAIN 유지 (그대로 둠)
    for img in train_imgs:
        img_path = os.path.join(IMAGE_DIR, img)
        label_path = os.path.join(LABEL_DIR, img.replace(".jpg", ".txt"))

        shutil.copy(img_path, os.path.join(OUT_IMAGE_TRAIN, img))

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUT_LABEL_TRAIN, img.replace(".jpg", ".txt")))

    # VAL 이동
    for img in val_imgs:
        img_path = os.path.join(IMAGE_DIR, img)
        label_path = os.path.join(LABEL_DIR, img.replace(".jpg", ".txt"))

        shutil.copy(img_path, os.path.join(OUT_IMAGE_VAL, img))

        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUT_LABEL_VAL, img.replace(".jpg", ".txt")))

    print("DONE: safe split complete")


if __name__ == "__main__":
    split_dataset()