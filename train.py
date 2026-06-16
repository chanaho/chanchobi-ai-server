from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",

    # 🔥 핵심 성능 설정
    epochs=120,
    imgsz=640,
    batch=8,

    # augmentation (중요)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,

    # 안정화
    patience=30,
    cos_lr=True,
    optimizer="AdamW",

    # 결과 저장
    project="runs/detect",
    name="farm_best_model",
)