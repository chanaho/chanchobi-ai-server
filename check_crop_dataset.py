import os

base = "dataset_crop"

print("\n===== CLASSIFY DATASET CHECK =====")

for split in ["train", "val"]:
    path = os.path.join(base, split)

    print(f"\n📁 {split}:", path)
    print("존재:", os.path.exists(path))

    if os.path.exists(path):
        classes = os.listdir(path)
        print("클래스:", classes)

        for c in classes:
            img_path = os.path.join(path, c)
            if os.path.isdir(img_path):
                print(f" - {c}: {len(os.listdir(img_path))}개 이미지")