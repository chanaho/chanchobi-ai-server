import os
import json
import shutil
import random


SRC_IMG = r"D:\앱개발\식물병 유발\원천데이터\TS1_고추_병해1\고추\병해"

SRC_JSON = r"D:\앱개발\식물병 유발\라벨링데이터\TL1_고추\병해"


OUT = "dataset_aihub_pepper"


os.makedirs(
    OUT+"/images/train",
    exist_ok=True
)

os.makedirs(
    OUT+"/images/val",
    exist_ok=True
)

os.makedirs(
    OUT+"/labels/train",
    exist_ok=True
)

os.makedirs(
    OUT+"/labels/val",
    exist_ok=True
)


files=os.listdir(SRC_JSON)


random.shuffle(files)


split=int(len(files)*0.8)


train_files=files[:split]
val_files=files[split:]


def convert(files, mode):

    for jf in files:

        json_file=os.path.join(
            SRC_JSON,
            jf
        )


        with open(
            json_file,
            encoding="utf-8"
        ) as f:
            data=json.load(f)


        img_name=data["description"]["image"]

        img_path=os.path.join(
            SRC_IMG,
            img_name
        )


        if not os.path.exists(img_path):
            continue


        width=data["description"]["width"]
        height=data["description"]["height"]


        bbox=data["annotations"]["bbox"][0]


        x=bbox["x"]
        y=bbox["y"]
        w=bbox["w"]
        h=bbox["h"]


        x_center=(x+w/2)/width
        y_center=(y+h/2)/height

        nw=w/width
        nh=h/height


        txt=img_name.replace(
            ".JPG",
            ".txt"
        )


        label_path=os.path.join(
            OUT,
            "labels",
            mode,
            txt
        )


        with open(
            label_path,
            "w"
        ) as f:

            # 0 = 고추 병해
            f.write(
                f"0 {x_center} {y_center} {nw} {nh}"
            )


        shutil.copy(
            img_path,
            os.path.join(
                OUT,
                "images",
                mode,
                img_name
            )
        )


convert(
    train_files,
    "train"
)

convert(
    val_files,
    "val"
)


print("완료")