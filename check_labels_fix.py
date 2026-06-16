import os

label_dirs = [
    "dataset_clean/labels/train",
    "dataset_clean/labels/val"
]

empty = 0
bad = 0
good = 0

for dir_path in label_dirs:
    if not os.path.exists(dir_path):
        continue

    for f in os.listdir(dir_path):
        if not f.endswith(".txt"):
            continue

        path = os.path.join(dir_path, f)

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if len(lines) == 0:
            empty += 1
            continue

        ok = True
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                ok = False
            else:
                try:
                    cls = int(parts[0])
                except:
                    ok = False

        if ok:
            good += 1
        else:
            bad += 1

print("\n===== LABEL 검사 결과 =====")
print("정상:", good)
print("비정상:", bad)
print("빈 파일:", empty)