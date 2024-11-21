from ultralytics import YOLO
import cv2

model_path = "runs/detect/train3/weights/last.pt"
model = YOLO(model_path)


image_path = "test/images/image2.jpeg"
image = cv2.imread(image_path)


results = model.predict(source=image_path, conf=0.14, show=True)
print("Results:", results)

highest = 0
labelHigh = ''

for result in results:
    for box in result.boxes:
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        label = model.names[class_id]
        print(f"Detected {label} with confidence: {confidence:.2f}")
        if float(f'{confidence:.2}') > highest:
            highest = float(f'{confidence:.2}')
            labelHigh = label

print(labelHigh,':',highest)
