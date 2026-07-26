import os
import shutil
import random

src = "my_farm_photos"
dst = "dataset_chanchobi_cls_v1"

train_ratio = 0.8

random.seed(42)

# 기존 폴더 생성
for split in ["train", "val"]:
    os.makedirs(
        os.path.join(dst, split),
        exist_ok=True
    )


total = 0


for crop in os.listdir(src):

    crop_path = os.path.join(src, crop)

    if not os.path.isdir(crop_path):
        continue

    for disease in os.listdir(crop_path):

        disease_path = os.path.join(
            crop_path,
            disease
        )

        if not os.path.isdir(disease_path):
            continue


        images = []

        for root, dirs, files in os.walk(disease_path):

            for f in files:

                if f.lower().endswith(
                    (".jpg",".jpeg",".png")
                ):
                    images.append(
                        os.path.join(root,f)
                    )


        if len(images)==0:
            continue


        class_name = crop + "_" + disease


        random.shuffle(images)

        split = int(len(images)*train_ratio)

        train_files = images[:split]
        val_files = images[split:]


        for name, files in [
            ("train", train_files),
            ("val", val_files)
        ]:

            out_dir = os.path.join(
                dst,
                name,
                class_name
            )

            os.makedirs(
                out_dir,
                exist_ok=True
            )


            for img in files:

                shutil.copy2(
                    img,
                    out_dir
                )

                total += 1


        print(
            class_name,
            "train:",
            len(train_files),
            "val:",
            len(val_files)
        )


print("====================")
print("총 이미지:", total)