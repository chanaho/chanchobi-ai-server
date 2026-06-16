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

files = [
    f for f in os.listdir(SOURCE)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print("총 이미지:", len(files))

for f in files:
    src = os.path.join(SOURCE, f)

    # 🔥 파일명 기준 라벨 판단
    label = None
    for c in CLASSES:
        if c in f:
            label = c
            break

    if label is None:
        print("스킵:", f)
        continue

    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, f)
    else:
        dst = os.path.join(TARGET, "val", label, f)

    shutil.copy(src, dst)

print("✔ dataset_crop 생성 완료")