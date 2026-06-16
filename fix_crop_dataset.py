import os
import shutil

src = "dataset_clean/images/train"
dst = "dataset_crop/train"

os.makedirs(dst, exist_ok=True)

for file in os.listdir(src):
    if not file.endswith(".jpg"):
        continue

    label = file.split("_")[0]

    out_dir = os.path.join(dst, label)
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy(os.path.join(src, file), os.path.join(out_dir, file))

print("DONE")