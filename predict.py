from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
model_path = "runs/detect/train2/weights/last.pt"  # Path to the trained YOLOv8 model
model = YOLO(model_path)

# Load an image for prediction
image_path = "test/images/image1.jpg"  # Path to the image you want to predict
image = cv2.imread(image_path)

# Run prediction with a lower confidence threshold
results = model.predict(source=image_path, conf=0.02, show=True)

# Print results for debugging
print("Results:", results)

# Draw bounding boxes on the image and print confidence levels
for result in results:
    for box in result.boxes:
        # Get bounding box coordinates and other details
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        confidence = box.conf[0]  # Confidence of the detection
        class_id = int(box.cls[0])  # Class ID
        label = model.names[class_id]  # Get class name from model

        # Print confidence level in the console
        print(f"Detected {label} with confidence: {confidence:.2f}")

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with predictions
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
