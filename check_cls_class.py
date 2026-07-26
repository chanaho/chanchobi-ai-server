import os

base = "dataset_chanchobi_cls_v1"

for split in ["train", "val"]:

    path = os.path.join(base, split)

    print("\n====================")
    print(split)

    total = 0

    for cls in sorted(os.listdir(path)):

        cls_path = os.path.join(path, cls)

        if not os.path.isdir(cls_path):
            continue

        count = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith(
                (".jpg",".jpeg",".png")
            )
        ])

        total += count

        print(
            f"{cls:35s} : {count}"
        )

    print("--------------------")
    print("전체:", total)