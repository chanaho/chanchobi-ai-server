import os

DATASET_PATH = "dataset_crop/val"

print("\n===== VAL DATASET COUNT =====\n")

for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)

    if os.path.isdir(cls_path):
        count = len(os.listdir(cls_path))
        print(f"{cls} : {count}")