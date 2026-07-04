"""
camera_stream.py
----------------
Uses YOLOv8 (via ultralytics) to detect people and vehicles live from
the webcam. Draws an orange box with the object name + confidence
(e.g. "person 84%") around each detection we care about, and logs
detections every 5 seconds so the terminal doesn't get spammed.

Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO
from datetime import datetime

MODEL_NAME = "yolov8n.pt"  # Nano model: fastest, good enough for real-time use
CONFIDENCE_THRESHOLD = 0.5
LOG_COOLDOWN_SECONDS = 5

# Only these COCO classes matter for a security camera
TARGET_CLASSES = {"person", "car", "truck", "motorcycle", "bus"}

BOX_COLOR = (0, 165, 255)  # orange in BGR


def stream():
    print("[INFO] Loading YOLOv8 model (first run may download weights)...")
    model = YOLO(MODEL_NAME)

    print("[INFO] Starting camera...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Could not open camera.")
        return

    last_log_time = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        results = model(frame, verbose=False)[0]

        detections_this_frame = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in TARGET_CLASSES:
                continue
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label = f"{class_name} {confidence * 100:.0f}%"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

            detections_this_frame.append(label)

        if detections_this_frame:
            now = datetime.now()
            if last_log_time is None or (now - last_log_time).total_seconds() >= LOG_COOLDOWN_SECONDS:
                timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DETECTED] {timestamp_str} — {', '.join(detections_this_frame)}")
                last_log_time = now

        cv2.imshow("YOLO Human & Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream()