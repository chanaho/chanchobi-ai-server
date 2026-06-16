from collections import Counter
from pathlib import Path

LABEL_DIR = Path("dataset/labels")

def read_labels(split):
    files = list((LABEL_DIR / split).glob("*.txt"))
    counter = Counter()

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            cls = int(line.split()[0])
            counter[cls] += 1

    return counter

def print_stats():
    train = read_labels("train")
    val = read_labels("val")

    print("\n📊 TRAIN 분포")
    for k, v in train.items():
        print(k, v)

    print("\n📊 VAL 분포")
    for k, v in val.items():
        print(k, v)

    print("\n⚠️ imbalance 체크")
    for k in train:
        ratio = val.get(k, 0) / max(train[k], 1)
        print(f"class {k}: val/train = {ratio:.3f}")

if __name__ == "__main__":
    print_stats()