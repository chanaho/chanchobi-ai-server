import os
import shutil
import random

SOURCE = "my_farm_photos"
TARGET = "dataset_crop"

CLASSES = ["사과", "자두", "고추", "복숭아", "수박", "아로니아", "오이", "포도"]

TRAIN_RATIO = 0.8

# 초기화
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

files = [
    f for f in os.listdir(SOURCE)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print("🔥 원본 이미지:", len(files))

train_count = 0
val_count = 0

for f in files:
    label = None

    # 파일명 기준 분류
    for c in CLASSES:
        if c in f:
            label = c
            break

    if label is None:
        continue

    src = os.path.join(SOURCE, f)

    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, f)
        train_count += 1
    else:
        dst = os.path.join(TARGET, "val", label, f)
        val_count += 1

    shutil.copy(src, dst)

print("\n✔ train:", train_count)
print("✔ val:", val_count)
print("🚀 dataset_crop 생성 완료")