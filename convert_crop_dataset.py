import os
import shutil

src = "dataset_clean/images/train"
dst = "dataset_crop/train"

os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if not f.endswith(".jpg"):
        continue

    label = f.split("_")[0]  # 사과_정상.jpg → 사과
    target_dir = os.path.join(dst, label)

    os.makedirs(target_dir, exist_ok=True)

    shutil.copy(
        os.path.join(src, f),
        os.path.join(target_dir, f)
    )

print("DONE")