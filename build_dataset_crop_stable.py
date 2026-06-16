import os
import shutil
import random

SOURCE = "dataset_clean/images"
TARGET = "dataset_crop"

CLASSES = ["사과", "자두", "고추", "복숭아", "수박", "아로니아", "오이", "포도"]

TRAIN_RATIO = 0.8

# 초기화
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

# 폴더 생성
for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

def get_label(filename):
    for c in CLASSES:
        if c in filename:
            return c
    return None

all_files = []

for split in ["train", "val"]:
    path = os.path.join(SOURCE, split)

    for f in os.listdir(path):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            all_files.append((split, f))

print("총 이미지:", len(all_files))

train_count = 0
val_count = 0

for split, f in all_files:
    label = get_label(f)

    if label is None:
        continue

    src = os.path.join(SOURCE, split, f)

    # 🔥 강제 train/val 분배 (균형 보장)
    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, f)
        train_count += 1
    else:
        dst = os.path.join(TARGET, "val", label, f)
        val_count += 1

    shutil.copy(src, dst)

print("✔ train:", train_count)
print("✔ val:", val_count)
print("🚀 완료")