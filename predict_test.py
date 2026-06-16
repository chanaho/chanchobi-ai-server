from ultralytics import YOLO

model = YOLO("runs/detect/final_stable/weights/best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True,
    imgsz=640
)

print("DONE")