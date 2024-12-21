# Grocery Detection Model using YOLOv8

## Overview
This project is a custom-trained object detection model using YOLOv8 to identify grocery items such as apples, tomatoes, bananas, and grapes. The model is trained on a labeled dataset and uses OpenCV for image processing.

## Features
- Custom YOLOv8 model for grocery detection.
- Supports real-time prediction with confidence thresholding.
- Identifies four types of objects: apple, tomato, banana, and grapes.
- Outputs predictions with labels and confidence scores.

---

## Directory Structure
```
project-directory/
├── config.yaml
├── predict.py
├── test/
│   └── images/
│       └── image2.jpeg
├── runs/
│   └── detect/
│       └── train3/
│           └── weights/
│               └── last.pt
└── data/
    └── images/
        ├── train/
        └── val/
```

---

## Dataset Configuration (config.yaml)
```
path: C:\Users\billi\Downloads\groc_dectector\data\
train: images/train
val: images/train
names:
  0: apple
  1: tomato
  2: banana
  3: grapes
```
- **path**: Path to the dataset folder.
- **train**: Training images location.
- **val**: Validation images location.
- **names**: Class labels mapped to indices.

---

### Script Breakdown
1. **Load Model:** Loads the trained YOLOv8 model.
2. **Read Image:** Loads an image for testing.
3. **Predict Objects:** Detects objects in the image based on confidence threshold (0.14).
4. **Display Results:** Shows the detection results and the label with the highest confidence.

---

## Requirements
- Python 3.8+
- ultralytics==8.0.0 (or later)
- OpenCV

### Install Dependencies
```bash
pip install ultralytics opencv-python
```

---

## Running the Model
1. Place test images in the `test/images/` directory.
2. Run the script:
```bash
python predict.py
```
3. View detected objects and confidence levels.

---

## Steps to Start Training
1. Prepare the dataset in the `data/images/train` and `data/images/val` directories.
2. Update the `config.yaml` file with the correct dataset paths and class names.
3. Install dependencies:
```bash
pip install ultralytics opencv-python
```
4. Initialize the model and start training:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='config.yaml', epochs=50, imgsz=640)
```
5. Monitor training progress and performance metrics.
6. Save the best-performing model weights located in the `runs/detect/trainX/weights/` directory.

---

## Results
- Outputs detected objects with confidence levels.
- Displays label with the highest confidence.

---

## Notes
- Ensure the dataset is properly labeled and organized.
- Adjust confidence threshold (`conf=0.6`) as needed for performance.
