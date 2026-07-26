import os


base="dataset_chanchobi_cls_v2"


for split in ["train","val"]:

    print("\n================")
    print(split)

    total=0

    path=os.path.join(base,split)

    for cls in sorted(os.listdir(path)):

        cls_path=os.path.join(path,cls)

        if os.path.isdir(cls_path):

            count=len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(
                    (".jpg",".jpeg",".png")
                )
            ])

            total += count

            print(
                f"{cls:25s}: {count}"
            )

    print("----------------")
    print("총 이미지:",total)