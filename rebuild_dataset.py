import os
import shutil
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = "."
OUT_DIR = "dataset_clean"

TRAIN_RATIO = 0.8

IMG_EXT = [".jpg", ".jpeg", ".png"]

# =========================
# CREATE FOLDERS
# =========================
def make_dirs():
    for p in [
        f"{OUT_DIR}/images/train",
        f"{OUT_DIR}/images/val",
        f"{OUT_DIR}/labels/train",
        f"{OUT_DIR}/labels/val",
    ]:
        os.makedirs(p, exist_ok=True)

# =========================
# GET FILES
# =========================
def get_images():
    imgs = []
    for root, _, files in os.walk(BASE_DIR):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXT:
                imgs.append(os.path.join(root, f))
    return imgs

# =========================
# SAFE COPY
# =========================
def copy_pair(img_path, split):
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + ".txt"

    label_path = img_path.replace("images", "labels").replace(img_name, label_name)

    out_img_dir = f"{OUT_DIR}/images/{split}"
    out_lbl_dir = f"{OUT_DIR}/labels/{split}"

    shutil.copy2(img_path, f"{out_img_dir}/{img_name}")

    # label 존재하면 복사, 없으면 empty label 생성
    if os.path.exists(label_path):
        shutil.copy2(label_path, f"{out_lbl_dir}/{label_name}")
    else:
        open(f"{out_lbl_dir}/{label_name}", "w").close()

# =========================
# MAIN
# =========================
def main():
    make_dirs()

    images = get_images()
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    print(f"TOTAL: {len(images)}")
    print(f"TRAIN: {len(train_imgs)}")
    print(f"VAL: {len(val_imgs)}")

    for img in train_imgs:
        copy_pair(img, "train")

    for img in val_imgs:
        copy_pair(img, "val")

    print("DONE: dataset_clean created safely")

if __name__ == "__main__":
    main()