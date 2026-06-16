import os
import shutil
from collections import Counter

DATASET = "dataset_clean"

IMG_TRAIN = f"{DATASET}/images/train"
LBL_TRAIN = f"{DATASET}/labels/train"

TARGET_MULTIPLIER = 5

def read_labels():
    counter = Counter()

    for f in os.listdir(LBL_TRAIN):
        if not f.endswith(".txt"):
            continue

        path = os.path.join(LBL_TRAIN, f)

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                try:
                    cls = int(parts[0])
                    counter[cls] += 1
                except:
                    pass

    return counter

def boost():
    counts = read_labels()
    if not counts:
        print("NO LABELS")
        return

    min_class = min(counts, key=counts.get)
    print("MIN CLASS:", min_class)

    imgs = os.listdir(IMG_TRAIN)
    added = 0

    for img in imgs:
        name = os.path.splitext(img)[0]
        lbl = name + ".txt"

        img_path = os.path.join(IMG_TRAIN, img)
        lbl_path = os.path.join(LBL_TRAIN, lbl)

        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            content = f.read()

        if str(min_class) in content:
            new_img = f"aug_{added}_{img}"
            new_lbl = f"aug_{added}_{lbl}"

            shutil.copy2(img_path, os.path.join(IMG_TRAIN, new_img))
            shutil.copy2(lbl_path, os.path.join(LBL_TRAIN, new_lbl))

            added += 1

        if added > len(imgs) * TARGET_MULTIPLIER:
            break

    print("DONE:", added)

if __name__ == "__main__":
    boost()