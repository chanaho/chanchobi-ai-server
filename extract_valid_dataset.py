import os
import shutil

src_img = "dataset_clean/images/train"
src_lbl = "dataset_clean/labels/train"

dst_img = "dataset_clean_filtered/images/train"
dst_lbl = "dataset_clean_filtered/labels/train"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

labels = [f for f in os.listdir(src_lbl) if f.endswith(".txt")]

copied = 0

for f in labels:
    lbl_path = os.path.join(src_lbl, f)

    with open(lbl_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 🚨 빈 라벨 제외
    if len(lines) == 0:
        continue

    img_name = f.replace(".txt", ".jpg")
    img_path = os.path.join(src_img, img_name)

    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(dst_img, img_name))
        shutil.copy(lbl_path, os.path.join(dst_lbl, f))
        copied += 1

print("\n===== 결과 =====")
print("사용 가능한 학습 데이터:", copied)