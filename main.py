from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.yaml")
results = model.train(data="config.yaml", epochs=50)
