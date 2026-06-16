import os
import shutil
import random

SOURCE = "dataset_clean"  # 원본 이미지 폴더
TARGET = "dataset_crop"

CLASSES = ["사과", "자두", "고추", "복숭아", "수박", "아로니아", "오이", "포도"]

# 비율
TRAIN_RATIO = 0.8

# 생성
for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

# 분류
for file in os.listdir(SOURCE):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    # 라벨 추출 (파일명 기준)
    label = file.split("_")[0]

    if label not in CLASSES:
        continue

    src = os.path.join(SOURCE, file)

    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, file)
    else:
        dst = os.path.join(TARGET, "val", label, file)

    shutil.copy(src, dst)

print("✔ dataset_crop 재구성 완료")