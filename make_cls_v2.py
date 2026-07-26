import os
import shutil

src = "dataset_chanchobi_cls_v1"
dst = "dataset_chanchobi_cls_v2"


classes = [

    "자두_정상",
    "자두_병징",
    "자두_잉크병",
    "자두_진딧물",

    "아로니아_정상",
    "아로니아_병징",
    "아로니아_진딧물",

    "사과_정상",
    "사과_병징",
    "사과_탄저병",

    "한라봉_정상",
    "한라봉_병징",

    "블루베리_정상",
    "블루베리_병징",

    "블랙커런트_정상",
    "블랙커런트_병징",

    "고추_정상",
    "고추_탄저병"
]


for split in ["train", "val"]:

    for cls in classes:

        src_dir = os.path.join(
            src,
            split,
            cls
        )

        dst_dir = os.path.join(
            dst,
            split,
            cls
        )

        if not os.path.exists(src_dir):
            continue


        os.makedirs(
            dst_dir,
            exist_ok=True
        )


        files = os.listdir(src_dir)


        for f in files:

            if f.lower().endswith(
                (".jpg",".jpeg",".png")
            ):

                shutil.copy2(
                    os.path.join(src_dir,f),
                    os.path.join(dst_dir,f)
                )


        print(
            split,
            cls,
            ":",
            len(files)
        )


print("===================")
print("완료")