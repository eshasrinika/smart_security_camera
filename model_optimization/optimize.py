"""
optimize.py
-----------
Benchmarks two YOLOv8 model sizes (Nano and Small) on YOUR laptop to
see which one actually runs faster. Captures one real frame from the
webcam, runs each model on it 10 times, measures average inference
time, converts that to FPS (frames per second), and saves a comparison
report to reports/model_comparison.txt.

Real security cameras (especially on small devices like Raspberry Pi)
need this kind of test to pick the best speed/accuracy tradeoff.
"""

import os
import time
import cv2
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
REPORT_FILE = os.path.join(REPORTS_DIR, "model_comparison.txt")

MODELS_TO_TEST = {
    "YOLOv8 Nano": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
}

NUM_RUNS = 10


def capture_test_frame():
    print("[INFO] Starting camera to capture a test frame...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        raise RuntimeError("Could not open camera.")

    # give the camera a moment to warm up / auto-adjust exposure
    for _ in range(5):
        video_capture.read()

    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        raise RuntimeError("Failed to capture a test frame.")

    print("[INFO] Test frame captured.")
    return frame


def benchmark_model(model_name, weights_file, test_frame):
    print(f"\n[INFO] Loading {model_name} ({weights_file})...")
    model = YOLO(weights_file)

    print(f"[INFO] Running {model_name} {NUM_RUNS} times to warm up and measure speed...")

    # one "warm-up" run — first inference is always slower (model initialization),
    # so we don't want that skewing the average
    model(test_frame, verbose=False)

    durations = []
    for i in range(NUM_RUNS):
        start_time = time.time()
        model(test_frame, verbose=False)
        end_time = time.time()
        durations.append(end_time - start_time)
        print(f"  Run {i + 1}/{NUM_RUNS}: {durations[-1] * 1000:.1f} ms")

    average_seconds = sum(durations) / len(durations)
    fps = 1 / average_seconds if average_seconds > 0 else 0

    return average_seconds, fps


def run_optimization():
    test_frame = capture_test_frame()

    results = []

    for model_name, weights_file in MODELS_TO_TEST.items():
        avg_seconds, fps = benchmark_model(model_name, weights_file, test_frame)
        results.append((model_name, weights_file, avg_seconds, fps))

    # sort fastest first
    results.sort(key=lambda r: r[3], reverse=True)

    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    for name, weights, avg_seconds, fps in results:
        print(f"{name:15s} | Avg time: {avg_seconds * 1000:6.1f} ms | FPS: {fps:5.1f}")

    winner_name, winner_weights, winner_seconds, winner_fps = results[0]
    print(f"\n[RECOMMENDATION] {winner_name} is the fastest on this laptop ({winner_fps:.1f} FPS).")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(REPORT_FILE, "w") as f:
        f.write("YOLOv8 Model Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        for name, weights, avg_seconds, fps in results:
            f.write(f"{name} ({weights})\n")
            f.write(f"  Average inference time: {avg_seconds * 1000:.1f} ms\n")
            f.write(f"  FPS: {fps:.1f}\n\n")
        f.write(f"Recommendation: {winner_name} is the fastest on this device ")
        f.write(f"({winner_fps:.1f} FPS), and is best suited for real-time security use.\n")

    print(f"\n[INFO] Report saved to {REPORT_FILE}")


if __name__ == "__main__":
    run_optimization()