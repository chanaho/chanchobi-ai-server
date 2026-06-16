import os
import shutil
import random

SOURCE = r"C:\chanchobi-app\ai-server\my_farm_photos"
TARGET = "dataset_crop"

TRAIN_RATIO = 0.8

# 안전 체크
if not os.path.exists(SOURCE):
    raise Exception(f"원본 폴더 없음: {SOURCE}")

# 초기화
if os.path.exists(TARGET):
    shutil.rmtree(TARGET)

# 클래스 = 폴더 이름
CLASSES = [
    d for d in os.listdir(SOURCE)
    if os.path.isdir(os.path.join(SOURCE, d))
]

print("🔥 클래스 발견:", CLASSES)

for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(TARGET, split, c), exist_ok=True)

train_count = 0
val_count = 0

for cls in CLASSES:
    path = os.path.join(SOURCE, cls)

    files = [
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"📦 {cls}: {len(files)}장")

    for f in files:
        src = os.path.join(path, f)

        if random.random() < TRAIN_RATIO:
            dst = os.path.join(TARGET, "train", cls, f)
            train_count += 1
        else:
            dst = os.path.join(TARGET, "val", cls, f)
            val_count += 1

        shutil.copy(src, dst)

print("\n✔ train:", train_count)
print("✔ val:", val_count)
print("🚀 dataset_crop 완료")