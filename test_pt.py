from ultralytics import YOLO

model = YOLO("runs/detect/train-13/weights/best.pt")

results = model.predict(
    source="aronia_botrytis1.jpg",
    imgsz=320,
    conf=0.001,
    verbose=True
)

r = results[0]

print("Boxes:", len(r.boxes))

if len(r.boxes):
    print("Classes:", r.boxes.cls.tolist())
    print("Conf:", r.boxes.conf.tolist())