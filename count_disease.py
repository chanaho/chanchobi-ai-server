from collections import Counter
import os

counter = Counter()

with open("disease_photos.txt", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        name = os.path.basename(line.strip())

        if "_" in name:
            parts = name.split("_")

            if len(parts) >= 2:
                key = parts[0] + "_" + parts[1].split(".")[0]
                counter[key] += 1

print("\n===== 병명 집계 =====\n")

for k, v in counter.most_common():
    print(f"{k}: {v}")