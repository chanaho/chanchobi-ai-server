import os

LABEL_DIR = "dataset/labels/train"

# 기존 disease → aphid로 임시 매핑
OLD_TO_NEW = {
    "0": "0"  # disease → aphid (임시)
}

def convert_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) == 0:
            continue

        cls = parts[0]

        if cls in OLD_TO_NEW:
            parts[0] = OLD_TO_NEW[cls]
            new_lines.append(" ".join(parts) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("변환 완료:", path)


def main():
    for file in os.listdir(LABEL_DIR):
        if file.endswith(".txt"):
            convert_file(os.path.join(LABEL_DIR, file))

    print("전체 변환 완료")


if __name__ == "__main__":
    main()