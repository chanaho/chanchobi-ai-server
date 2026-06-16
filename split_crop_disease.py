import os
import shutil

src_img = "dataset_clean/images/train"
src_lbl = "dataset_clean/labels/train"

crop_dst_img = "dataset_crop/images/train"
crop_dst_lbl = "dataset_crop/labels/train"

disease_dst_img = "dataset_disease/images/train"
disease_dst_lbl = "dataset_disease/labels/train"

os.makedirs(crop_dst_img, exist_ok=True)
os.makedirs(crop_dst_lbl, exist_ok=True)
os.makedirs(disease_dst_img, exist_ok=True)
os.makedirs(disease_dst_lbl, exist_ok=True)

crop_keywords = ["사과", "자두", "블루베리", "복숭아", "대추"]

count = 0

for f in os.listdir(src_lbl):
    if not f.endswith(".txt"):
        continue

    path = os.path.join(src_lbl, f)

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if len(lines) == 0:
        continue

    img_name = f.replace(".txt", ".jpg")
    img_path = os.path.join(src_img, img_name)

    if not os.path.exists(img_path):
        continue

    text = "".join(lines)

    # 🌱 작물 데이터
    if any(k in text for k in crop_keywords):
        shutil.copy(img_path, os.path.join(crop_dst_img, img_name))
        shutil.copy(path, os.path.join(crop_dst_lbl, f))

    # 🦠 병 데이터 (나머지)
    else:
        shutil.copy(img_path, os.path.join(disease_dst_img, img_name))
        shutil.copy(path, os.path.join(disease_dst_lbl, f))

    count += 1

print("\n===== 완료 =====")
print("분류된 이미지:", count)