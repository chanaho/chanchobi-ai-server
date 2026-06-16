import os

cmd = """
yolo detect train ^
model=yolov8n.pt ^
data=crop_dataset.yaml ^
epochs=200 ^
imgsz=640 ^
batch=16 ^
optimizer=auto ^
lr0=0.001 ^
close_mosaic=10 ^
mosaic=1.0 ^
mixup=0.1 ^
copy_paste=0.1 ^
project=runs/detect ^
name=final_agri_run
"""

os.system(cmd)