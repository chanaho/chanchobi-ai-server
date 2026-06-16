from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="my_farm_photos",
    epochs=30,
    imgsz=224,
    batch=16
)