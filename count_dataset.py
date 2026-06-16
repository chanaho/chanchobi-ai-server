import os

DATASET_PATH = "dataset_crop/train"

print("\n===== DATASET COUNT =====\n")

total = 0

for cls in sorted(os.listdir(DATASET_PATH)):
    cls_path = os.path.join(DATASET_PATH, cls)

    if os.path.isdir(cls_path):
        count = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

        total += count

        print(f"{cls} : {count}")

print("\n========================")
print("TOTAL :", total)