import os
import shutil
import random

IMG_ROOT = r"dataset_clean/images"
LABEL_ROOT = r"dataset_clean/labels"
TARGET = r"dataset_crop"

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

# 폴더 생성
for split in ["train", "val"]:
    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(TARGET, split, cls), exist_ok=True)

def process_split(split):
    img_dir = os.path.join(IMG_ROOT, split)
    label_dir = os.path.join(LABEL_ROOT, split)

    if not os.path.exists(img_dir):
        print(f"❌ 이미지 폴더 없음: {img_dir}")
        return

    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"\n🔥 {split} 이미지 수:", len(files))

    count = 0

    for img in files:
        label_file = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            continue

        try:
            cls_id = int(lines[0].split()[0])
        except:
            continue

        if cls_id not in CLASS_MAP:
            continue

        label = CLASS_MAP[cls_id]

        src = os.path.join(img_dir, img)

        if random.random() < TRAIN_RATIO:
            dst = os.path.join(TARGET, "train", label, img)
        else:
            dst = os.path.join(TARGET, "val", label, img)

        shutil.copy(src, dst)
        count += 1

    print(f"✔ {split} 완료:", count)

process_split("train")
process_split("val")

print("\n🚀 dataset_crop 최종 생성 완료")