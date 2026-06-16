from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset_crop",
    epochs=30,
    imgsz=224,
    batch=16,
    name="crop_v1"
)