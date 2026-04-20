from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,        # 基准尺寸
    multi_scale=False, # 多尺度训练
    scale =0.3, #目标缩放率
    mosaic = 1.0, # mosaic拼接增强的概率
    mixup = 0.1, # mixup融合增强的概率
    batch=32,
    workers=8
)