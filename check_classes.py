from pathlib import Path

root = Path("dataset_crop/train")

print("\n===== 클래스별 이미지 수 =====\n")

for cls in sorted(root.iterdir()):
    if cls.is_dir():
        count = len(list(cls.glob("*.jpg")))
        print(f"{cls.name} : {count}")