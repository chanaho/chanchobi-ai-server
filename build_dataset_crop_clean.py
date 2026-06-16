import os
import shutil
import random

SOURCE = "dataset_clean"
TARGET = "dataset_crop"

CLASSES = ["사과", "자두", "고추", "복숭아", "수박", "아로니아", "오이", "포도"]

TRAIN_RATIO = 0.8

# 초기화
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

# 구조 생성
for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

# 이미지 수집
files = [
    f for f in os.listdir(SOURCE)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# 분배
for f in files:
    label = f.split("_")[0]

    if label not in CLASSES:
        continue

    src = os.path.join(SOURCE, f)

    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, f)
    else:
        dst = os.path.join(TARGET, "val", label, f)

    shutil.copy(src, dst)

print("✔ dataset_crop 완전 재구성 완료")