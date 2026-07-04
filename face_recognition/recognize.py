"""
recognize.py
------------
Loads encodings.pickle (created by train.py) and opens your webcam.
Draws a GREEN box + name for known people, RED box + "Unknown" for
anyone not in your dataset. Press 'q' to quit.
"""

import face_recognition
import cv2
import pickle
import numpy as np
from datetime import datetime

ENCODINGS_FILE = "encodings.pickle"
TOLERANCE = 0.5            # lower = stricter matching (0.4-0.6 is typical)
FRAME_RESIZE_SCALE = 0.25  # shrink frame before processing, for speed


def load_known_faces():
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def recognize():
    print("[INFO] Loading known faces...")
    known_encodings, known_names = load_known_faces()

    print("[INFO] Starting camera...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Could not open camera.")
        return

    last_unknown_log_time = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            color = (0, 0, 255)  # red for unknown

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    color = (0, 255, 0)  # green for known

            # scale coordinates back up since we processed a shrunk frame
            top = int(top / FRAME_RESIZE_SCALE)
            right = int(right / FRAME_RESIZE_SCALE)
            bottom = int(bottom / FRAME_RESIZE_SCALE)
            left = int(left / FRAME_RESIZE_SCALE)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            if name == "Unknown":
                now = datetime.now()
                if last_unknown_log_time is None or (now - last_unknown_log_time).seconds >= 3:
                    print(f"[ALERT] Unknown person detected at {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    last_unknown_log_time = now

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition - Security System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
    