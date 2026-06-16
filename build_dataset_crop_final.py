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

for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

def process_split(split_name):
    path = os.path.join(SOURCE, split_name)

    files = [
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"\n🔥 {split_name} 이미지:", len(files))

    count = 0

    for f in files:
        label = None

        for c in CLASSES:
            if c in f:
                label = c
                break

        if label is None:
            continue

        src = os.path.join(path, f)

        dst = os.path.join(TARGET, split_name, label, f)

        shutil.copy(src, dst)
        count += 1

    print(f"✔ {split_name} 변환 완료:", count)

process_split("train")
process_split("val")

print("\n🚀 dataset_crop 최종 생성 완료")