"""
main.py
-------
The combined demo version of the security system. Runs three modules
together in a single camera window:

  - Face recognition   -> green box + name (known) / red box + "Unknown"
  - Motion detection    -> green "Motion Detected" outline
  - YOLO detection       -> orange box + label for person/car/truck/etc.

This is meant for demos/recordings — showing everything working
together in one place, rather than five separate windows.

Press 'q' to quit.
"""

import os
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(SCRIPT_DIR, "face_recognition", "encodings.pickle")

# --- Face recognition settings ---
FACE_TOLERANCE = 0.5
FRAME_RESIZE_SCALE = 0.25

# --- Motion detection settings ---
MIN_CONTOUR_AREA = 900
BACKGROUND_UPDATE_RATE = 0.02

# --- YOLO settings ---
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_TARGET_CLASSES = {"person", "car", "truck", "motorcycle", "bus"}
YOLO_BOX_COLOR = (0, 165, 255)  # orange


def load_known_faces():
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def run_face_recognition(frame, known_encodings, known_names):
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=FACE_TOLERANCE)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        color = (0, 0, 255)  # red

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                color = (0, 255, 0)  # green

        top = int(top / FRAME_RESIZE_SCALE)
        right = int(right / FRAME_RESIZE_SCALE)
        bottom = int(bottom / FRAME_RESIZE_SCALE)
        left = int(left / FRAME_RESIZE_SCALE)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


def run_motion_detection(frame, float_background):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    frame_delta = cv2.absdiff(cv2.convertScaleAbs(float_background), gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_found = False
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        motion_found = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if motion_found:
        cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # slowly adapt the background so lighting changes don't cause false alarms
    cv2.accumulateWeighted(gray_frame, float_background, BACKGROUND_UPDATE_RATE)


def run_yolo_detection(frame, model):
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name not in YOLO_TARGET_CLASSES or confidence < YOLO_CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), YOLO_BOX_COLOR, 2)
        label = f"{class_name} {confidence * 100:.0f}%"
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YOLO_BOX_COLOR, 2)


def main():
    print("[INFO] Loading known faces...")
    known_encodings, known_names = load_known_faces()

    print("[INFO] Loading YOLOv8 model...")
    yolo_model = YOLO(YOLO_MODEL)

    print("[INFO] Starting camera...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Could not open camera.")
        return

    ret, first_frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab the first frame.")
        return

    gray_background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.GaussianBlur(gray_background, (21, 21), 0)
    float_background = gray_background.astype("float")

    print("[INFO] All modules loaded. Starting combined demo — press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        run_motion_detection(frame, float_background)
        run_yolo_detection(frame, yolo_model)
        run_face_recognition(frame, known_encodings, known_names)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Smart Security Camera System - Combined Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()