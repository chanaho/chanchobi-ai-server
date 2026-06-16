import os

label_dirs = [
    "dataset_clean/labels/train",
    "dataset_clean/labels/val"
]

mapping = {
    0: 0,
    4: 1
}

print("\n===== 라벨 재매핑 시작 =====\n")

for base in label_dirs:
    if not os.path.exists(base):
        continue

    for f in os.listdir(base):
        if not f.endswith(".txt"):
            continue

        path = os.path.join(base, f)

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            cls = int(parts[0])

            if cls not in mapping:
                continue

            parts[0] = str(mapping[cls])
            new_lines.append(" ".join(parts))

        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(new_lines))

print("완료: class0, class4 → 0,1 변환")