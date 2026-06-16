import os
import random
import shutil

SOURCE_DIR = "dataset/raw"
TRAIN_DIR = "dataset/images/train"
VAL_DIR = "dataset/images/val"

SPLIT_RATE = 0.8

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)

    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATE)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    os.makedirs(
        os.path.join(TRAIN_DIR, cls),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(VAL_DIR, cls),
        exist_ok=True
    )

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(VAL_DIR, cls, img)
        )

print("데이터 분리 완료")