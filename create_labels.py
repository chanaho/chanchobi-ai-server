import os

def create_dummy_labels(image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    for img in os.listdir(image_dir):
        if img.endswith(".jpg") or img.endswith(".png"):
            txt_name = img.replace(".jpg", ".txt").replace(".png", ".txt")
            txt_path = os.path.join(label_dir, txt_name)

            with open(txt_path, "w") as f:
                # 클래스 0, 화면 중앙 박스 (YOLO 형식)
                f.write("0 0.5 0.5 0.5 0.5")

# 실행
create_dummy_labels("dataset/images/train", "dataset/labels/train")
create_dummy_labels("dataset/images/val", "dataset/labels/val")

print("라벨 생성 완료")