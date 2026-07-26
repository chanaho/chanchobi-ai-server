import os
import shutil
import glob


OUT = "dataset_chanchobi_final"


for folder in [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]:
    os.makedirs(
        os.path.join(OUT, folder),
        exist_ok=True
    )


def copy_dataset(src, name):

    print("\n====", name, "====")

    for split in ["train","val"]:

        img_dir = os.path.join(
            src,
            "images",
            split
        )

        label_dir = os.path.join(
            src,
            "labels",
            split
        )


        if not os.path.exists(img_dir):
            continue


        for img in glob.glob(
            img_dir+"/*"
        ):

            base=os.path.splitext(
                os.path.basename(img)
            )[0]


            label=os.path.join(
                label_dir,
                base+".txt"
            )


            # 라벨 없는 이미지는 제외
            if not os.path.exists(label):
                continue


            shutil.copy(
                img,
                OUT+"/images/"+split
            )


            shutil.copy(
                label,
                OUT+"/labels/"+split
            )


            print(
                split,
                base
            )


# 기존 데이터
copy_dataset(
    "dataset_clean",
    "OLD"
)


# AI HUB 고추
copy_dataset(
    "dataset_aihub_pepper",
    "AIHUB"
)


print("\n완료")