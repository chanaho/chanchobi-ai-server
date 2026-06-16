from ultralytics import YOLO

model = YOLO("runs/detect/final_stable/weights/best.pt")

model.predict(
    source=0,
    show=True,
    conf=0.25,
    imgsz=640
)