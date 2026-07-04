# AI-Powered Smart Security Camera System

A Python-based security camera system that goes beyond regular CCTV. Instead of just recording, it recognizes who's in front of the camera, notices when something moves, tells people apart from vehicles, and emails you a photo the moment a stranger shows up.

I built this as a way to combine computer vision and deep learning into something that actually does a job, rather than just being a notebook full of accuracy scores.

## Demo

(Add a link to your demo video or GIF here once it's recorded)

## What it does

- **Face recognition** — learns a set of known faces from photos and identifies them live through the webcam. Anyone it doesn't recognize gets flagged as "Unknown."
- **Motion detection** — watches for movement using background subtraction, so it only reacts when something actually changes in the frame rather than recording constantly.
- **Human and vehicle detection** — uses YOLOv8 to tell the difference between a person, a car, a truck, a motorcycle, or a bus, with a confidence score for each detection.
- **Automated alerts** — when an unknown face is detected, it saves a snapshot and emails it automatically, with a cooldown so it doesn't flood the inbox.
- **Model benchmarking** — times YOLOv8 Nano against YOLOv8 Small on the local machine to figure out which one is actually worth using for real-time detection.

## Tech stack

| Library | What it's doing here |
|---|---|
| OpenCV (`cv2`) | Handles the camera, frame processing, and all the on-screen drawing |
| `face_recognition` (built on `dlib`) | Detects faces and converts them into 128-number encodings for comparison |
| Ultralytics YOLOv8 | Real-time object detection for people and vehicles |
| PyTorch | The deep learning engine YOLO runs on |
| NumPy | Number-crunching for image arrays and face encodings |
| `smtplib` / `email` | Sends the alert emails through Gmail's SMTP server |

## Project structure

```
security_ai_project/
├── face_recognition/
│   ├── dataset/           # Training photos, one folder per person
│   ├── train.py           # Encodes faces from dataset/ into encodings.pickle
│   └── recognize.py       # Live webcam face recognition
├── motion_detection/
│   ├── detect.py          # Background-subtraction motion detection
│   └── alerts.py          # Throttled logging and snapshot saving
├── smart_camera/
│   └── camera_stream.py   # YOLOv8 human and vehicle detection
├── security_automation/
│   ├── automation.py      # Face recognition plus email alerts on unknown faces
│   └── config.py          # Email credentials (not committed, see below)
├── model_optimization/
│   └── optimize.py        # Benchmarks YOLOv8 Nano against Small
├── reports/                # Snapshots and benchmark output, generated when you run things
└── README.md
```

## Setup

1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd security_ai_project
   ```

2. Install the dependencies:
   ```bash
   pip install cmake dlib face_recognition opencv-python numpy ultralytics torch
   ```
   On Windows, if `dlib` refuses to build from source, try `pip install dlib-bin` instead — it's a precompiled version and usually saves the headache.

3. Add your own photos to `face_recognition/dataset/<name>/`. Three to five clear, well-lit photos per person is enough.

4. If you want the email alerts working, create `security_automation/config.py` with a Gmail address and a Google App Password (this requires 2-Step Verification to be turned on). That file is gitignored on purpose — don't commit real credentials to a public repo.

## Running it

Each module runs from inside its own folder:

```bash
# Train on your dataset photos (run once, or whenever you add new photos)
cd face_recognition
python train.py

# Live face recognition
python recognize.py

# Motion detection
cd ../motion_detection
python detect.py

# Human and vehicle detection
cd ../smart_camera
python camera_stream.py

# Email alerts for unknown faces
cd ../security_automation
python automation.py

# Benchmark model speed
cd ../model_optimization
python optimize.py
```

Press `q` in any camera window to close it.

## Benchmark results

| Model | Average inference time | FPS |
|---|---|---|
| YOLOv8 Nano | 128.9 ms | 7.8 |
| YOLOv8 Small | 258.1 ms | 3.9 |

Nano runs roughly twice as fast as Small on this machine, which makes it the better pick for real-time use on CPU-only hardware — the kind of tradeoff you'd actually have to make if this were running on something like a Raspberry Pi.

## Why it's built this way

This project was as much about the engineering decisions as the AI itself — figuring out how to keep alerts from spamming an inbox, how to avoid false motion triggers from lighting changes, and how to actually measure which model is worth using rather than just assuming the bigger one is better.

## Author

Bandaru Esha Srinika — Electronics and Communication Engineering graduate, GITAM University.