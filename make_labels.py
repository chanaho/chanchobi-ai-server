import os

# =========================
# 설정
# =========================
IMG_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

os.makedirs(LABEL_DIR, exist_ok=True)

# =========================
# 클래스 매핑
# =========================
class_map = {
    "고추_탄저병": 0,
    "사과_갈색무늬병": 1,
    "자두_세균성구멍병": 2,
    "아로니아_잿빛곰팡이병": 3,
    "정상": 4
}

# =========================
# 자동 라벨 생성
# =========================
for file in os.listdir(IMG_DIR):

    if not file.endswith(".jpg"):
        continue

    name = file.replace(".jpg", "")

    label_class = None

    # 이름 기준 클래스 찾기
    for k in class_map.keys():
        if k in name:
            label_class = class_map[k]
            break

    if label_class is None:
        print("❌ SKIP:", file)
        continue

    txt_path = os.path.join(
        LABEL_DIR,
        name + ".txt"
    )

    # YOLO 형식 (전체 이미지 기준)
    with open(txt_path, "w") as f:
        f.write(f"{label_class} 0.5 0.5 0.9 0.9\n")

    print("✅ CREATED:", txt_path)

print("🔥 DONE")