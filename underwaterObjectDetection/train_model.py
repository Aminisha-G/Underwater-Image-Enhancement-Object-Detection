from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="aquarium_pretrain/data.yaml",
    epochs=70,
    imgsz=768,
    batch=10,
)