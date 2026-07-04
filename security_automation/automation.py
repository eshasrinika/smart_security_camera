"""
automation.py
--------------
Runs face recognition on the webcam. Whenever an "Unknown" person is
detected, it saves a snapshot to reports/unknown_persons/ and sends
an email alert with that photo attached, via Gmail SMTP.

Waits ALERT_COOLDOWN_SECONDS between emails so you don't get spammed
with alerts if someone unknown lingers in frame.

Press 'q' to quit.
"""

import os
import cv2
import pickle
import smtplib
import numpy as np
import face_recognition
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from config import EMAIL_ADDRESS, EMAIL_APP_PASSWORD, RECEIVER_EMAIL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(SCRIPT_DIR, "..", "face_recognition", "encodings.pickle")
UNKNOWN_DIR = os.path.join(SCRIPT_DIR, "..", "reports", "unknown_persons")

TOLERANCE = 0.5
FRAME_RESIZE_SCALE = 0.25
ALERT_COOLDOWN_SECONDS = 30

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def load_known_faces():
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def send_alert_email(image_path, timestamp_str):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = "Security Alert - Unknown Person Detected"

        body = f"An unknown person was detected by your security camera at {timestamp_str}."
        msg.attach(MIMEText(body, "plain"))

        with open(image_path, "rb") as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(image_path))
        msg.attach(image)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
            server.send_message(msg)

        print(f"  [EMAIL] Alert sent to {RECEIVER_EMAIL}")

    except Exception as e:
        print(f"  [ERROR] Failed to send email: {e}")


def automate():
    print("[INFO] Loading known faces...")
    known_encodings, known_names = load_known_faces()

    os.makedirs(UNKNOWN_DIR, exist_ok=True)

    print("[INFO] Starting camera...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Could not open camera.")
        return

    last_alert_time = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        unknown_detected_this_frame = False

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
                unknown_detected_this_frame = True

        if unknown_detected_this_frame:
            now = datetime.now()
            if last_alert_time is None or (now - last_alert_time).total_seconds() >= ALERT_COOLDOWN_SECONDS:
                timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                filename = now.strftime("unknown_%Y%m%d_%H%M%S.jpg")
                filepath = os.path.join(UNKNOWN_DIR, filename)
                cv2.imwrite(filepath, frame)

                print(f"[ALERT] Unknown person detected at {timestamp_str}")
                print(f"  [SAVED] {filepath}")

                send_alert_email(filepath, timestamp_str)

                last_alert_time = now

        cv2.imshow("Security Automation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    automate()