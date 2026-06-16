import os

base = "dataset_clean"

paths = [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]

print("🔍 FINAL CHECK")

total = 0
for p in paths:
    full = os.path.join(base, p)
    count = len(os.listdir(full)) if os.path.exists(full) else 0
    print(p, "=>", count)
    total += count

print("\nTOTAL FILES:", total)