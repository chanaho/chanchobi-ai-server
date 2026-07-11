from ultralytics import YOLO

model = YOLO("model/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    simplify=True
)