from ultralytics import YOLO

model = YOLO("models/chanchobi_cls_best.pt")

model.export(
    format="onnx",
    imgsz=416,
    simplify=True
)