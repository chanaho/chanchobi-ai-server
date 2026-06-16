import os
import shutil
import random

IMG_DIR = "dataset_clean/images"
LABEL_DIR = "dataset_clean/labels"
TARGET = "dataset_crop"

CLASS_MAP = {
    0: "사과",
    1: "자두",
    2: "고추",
    3: "복숭아",
    4: "수박",
    5: "아로니아",
    6: "오이",
    7: "포도"
}

TRAIN_RATIO = 0.8

# 초기화
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

for split in ["train", "val"]:
    for c in CLASS_MAP.values():
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

print("🔥 이미지 수:", len(files))

count = 0

for img in files:
    label_file = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not lines:
        continue

    cls_id = int(lines[0].split()[0])

    if cls_id not in CLASS_MAP:
        continue

    label = CLASS_MAP[cls_id]

    src = os.path.join(IMG_DIR, img)

    if random.random() < TRAIN_RATIO:
        dst = os.path.join(TARGET, "train", label, img)
    else:
        dst = os.path.join(TARGET, "val", label, img)

    shutil.copy(src, dst)
    count += 1

print("\n✔ 최종 변환 완료:", count)