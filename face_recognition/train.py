"""
train.py
--------
Reads every photo inside dataset/<person_name>/, detects the face,
converts it into a 128-number "encoding", and saves all encodings +
names into encodings.pickle for recognize.py to use later.

Run this once (and again any time you add/change photos in dataset/).
"""

import face_recognition
import os
import pickle

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"


def train():
    known_encodings = []
    known_names = []

    print("[INFO] Starting training...")

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing images for: {person_name}")

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)

            face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 0:
                print(f"  [WARNING] No face found in {filename}, skipping.")
                continue

            if len(face_locations) > 1:
                print(f"  [WARNING] Multiple faces found in {filename}, using the first one.")

            encoding = face_recognition.face_encodings(
                image, known_face_locations=[face_locations[0]]
            )[0]

            known_encodings.append(encoding)
            known_names.append(person_name)
            print(f"  [OK] Encoded {filename}")

    data = {"encodings": known_encodings, "names": known_names}

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Training complete. {len(known_encodings)} face encodings saved to {ENCODINGS_FILE}")


if __name__ == "__main__":
    train()