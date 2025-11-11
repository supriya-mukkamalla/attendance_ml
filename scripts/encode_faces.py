# scripts/encode_faces.py
import face_recognition
import os
import pickle
import cv2

# Dataset folder (all student subfolders/images)
DATASET_DIR = "./dataset"

# Save encodings in project root
OUTPUT_FILE = "/Users/msupriya/attendance_ml/encodings.pickle"  # <-- absolute path

known_encodings = []
known_names = []

# Loop through each student folder
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue
    print("Processing:", person_name)
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print("WARNING: couldn't read", img_path)
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes = face_recognition.face_locations(rgb, model="hog")  # "hog" is faster
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 0:
            print("No faces found in", img_path)
            continue

        # Add each face encoding with the student name
        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(person_name)

# Save encodings to pickle
data = {"encodings": known_encodings, "names": known_names}
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print("✅ Encodings saved to", OUTPUT_FILE)
