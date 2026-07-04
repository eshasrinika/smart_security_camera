"""
detect.py
---------
Watches the webcam and detects motion using background subtraction.
Takes the first frame as the "empty room" baseline, then compares
every new frame against it. Draws a green box around anything that
changed enough to count as real motion, and logs it (throttled) via
alerts.py. The baseline slowly adapts so lighting changes don't cause
false alarms.

Press 'q' to quit.
"""

import cv2
from alerts import MotionLogger

MIN_CONTOUR_AREA = 900         # ignore tiny changes (noise, shadows)
BACKGROUND_UPDATE_RATE = 0.02  # how fast the baseline adapts to lighting changes


def detect():
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

    motion_logger = MotionLogger(save_snapshots=True)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # compare current frame against the adapted background
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion_found:
            cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            motion_logger.log(frame)

        # slowly blend the current frame into the background baseline
        cv2.accumulateWeighted(gray_frame, float_background, BACKGROUND_UPDATE_RATE)

        cv2.imshow("Motion Detection - Security System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()