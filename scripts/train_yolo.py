from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset_new/dataset.yaml",
    epochs=50,
    imgsz=640
)