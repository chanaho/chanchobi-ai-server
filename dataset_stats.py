import os
from collections import Counter

DATASET = "dataset_clean"

def count_split(split):
    path = f"{DATASET}/labels/{split}"
    counter = Counter()

    if not os.path.exists(path):
        print(split, "NOT FOUND")
        return counter

    for f in os.listdir(path):
        if not f.endswith(".txt"):
            continue

        with open(os.path.join(path, f), "r", encoding="utf-8") as file:
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

def main():
    train = count_split("train")
    val = count_split("val")

    print("TRAIN:", dict(train))
    print("VAL:", dict(val))

    total = train + val
    print("TOTAL:", dict(total))

if __name__ == "__main__":
    main()