import os
import shutil
import random

IMG_DIR = "dataset_clean/images"
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
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print("총 이미지:", len(files))

count = 0

for f in files:
    label = None

    # 🔥 파일명 기반 추출 (핵심)
    for c in CLASSES:
        if c in f:
            label = c
            break

    if label is None:
        continue

    src = os.path.join(IMG_DIR, f)

    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, f)
    else:
        dst = os.path.join(TARGET, "val", label, f)

    shutil.copy(src, dst)
    count += 1

print("✔ 변환 완료:", count)