import os
from collections import Counter

label_dir = "dataset_clean/labels"

counter = Counter()
empty = 0
total = 0

for split in ["train", "val"]:
    path = os.path.join(label_dir, split)

    if not os.path.exists(path):
        continue

    for f in os.listdir(path):
        if not f.endswith(".txt"):
            continue

        total += 1
        file_path = os.path.join(path, f)

        with open(file_path, "r") as file:
            lines = file.readlines()

        if len(lines) == 0:
            empty += 1
            continue

        for line in lines:
            cls = int(line.split()[0])
            counter[cls] += 1

print("📊 CLASS DISTRIBUTION:", dict(counter))
print("📦 TOTAL LABEL FILES:", total)
print("⚠️ EMPTY LABELS:", empty)