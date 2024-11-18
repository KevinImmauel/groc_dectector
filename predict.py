from ultralytics import YOLO
import cv2

model_path = "runs/detect/train/weights/last.pt"
model = YOLO(model_path)


image_path = "test/images/image1.jpg"
image = cv2.imread(image_path)


results = model.predict(source=image_path, conf=0.01, show=True)
print("Results:", results)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        label = model.names[class_id]

        print(f"Detected {label} with confidence: {confidence:.2f}")

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
