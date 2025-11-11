import csv
from datetime import datetime
import face_recognition
import pickle
import cv2
import os

# Path to encodings
ENCODINGS_FILE = "/Users/msupriya/attendance_ml/encodings.pickle"

# Absolute path to your group photo
GROUP_PHOTO = "/Users/msupriya/attendance_ml/group_photo.jpg"

# Load face encodings
data = pickle.load(open(ENCODINGS_FILE, "rb"))
known_encodings = data["encodings"]
known_names = data["names"]

# Load group image
image = cv2.imread(GROUP_PHOTO)
if image is None:
    raise ValueError(f"Cannot load image at {GROUP_PHOTO}. Check the path and filename!")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces using CNN model for higher accuracy
boxes = face_recognition.face_locations(rgb, model="cnn")
encodings = face_recognition.face_encodings(rgb, boxes)

present_students = []

for enc, box in zip(encodings, boxes):
    # Compare detected face with known encodings using tolerance 0.6
    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.6)
    name = "Unknown"
    if True in matches:
        matched_idxs = [i for i, m in enumerate(matches) if m]
        counts = {}
        for i in matched_idxs:
            counts[known_names[i]] = counts.get(known_names[i], 0) + 1
        name = max(counts, key=counts.get)
        present_students.append(name)
    
    # Draw rectangle and name on the image
    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Debug: print all detected names
print("Detected names:", present_students)

# Save the annotated image
cv2.imwrite("/Users/msupriya/attendance_ml/group_photo_annotated.jpg", image)
print("✅ Annotated photo saved as group_photo_annotated.jpg")

# Generate attendance list
all_students = os.listdir("/Users/msupriya/attendance_ml/dataset")
absent_students = [s for s in all_students if s not in present_students]

print("\n📋 Attendance List:")
print("Present:", present_students)
print("Absent:", absent_students)

# --- Save attendance to CSV ---
csv_file = "/Users/msupriya/attendance_ml/attendance.csv"

# Check if CSV already exists
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    # Write header if file did not exist
    if not file_exists:
        writer.writerow(["Date", "Student Name", "Status"])
    
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Mark present students
    for student in present_students:
        writer.writerow([today, student, "Present"])
    
    # Mark absent students
    for student in absent_students:
        writer.writerow([today, student, "Absent"])

print(f"✅ Attendance saved to {csv_file}")
