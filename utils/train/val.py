from ultralytics import YOLO

model = YOLO("/home/T7/zmj/机器臂实时抓取传送带货物系统/utils/train/runs/segment/train/weights/best.pt")

model.val(
    data="data.yaml",
    workers=8
)