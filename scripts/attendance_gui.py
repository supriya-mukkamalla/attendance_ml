import tkinter as tk
from tkinter import filedialog, messagebox
import face_recognition
import pickle
import cv2
import os
import csv
from datetime import datetime

# Paths
ENCODINGS_FILE = "/Users/msupriya/attendance_ml/encodings.pickle"
DATASET_DIR = "/Users/msupriya/attendance_ml/dataset"
CSV_FILE = "/Users/msupriya/attendance_ml/attendance.csv"

# Load known face encodings
data = pickle.load(open(ENCODINGS_FILE, "rb"))
known_encodings = data["encodings"]
known_names = data["names"]

def mark_attendance():
    # Ask teacher to select a group photo
    file_path = filedialog.askopenfilename(title="Select Group Photo",
                                           filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Load image
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Cannot open image. Check the file!")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    present_students = []

    for enc, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            matched_idxs = [i for i, m in enumerate(matches) if m]
            counts = {}
            for i in matched_idxs:
                counts[known_names[i]] = counts.get(known_names[i], 0) + 1
            name = max(counts, key=counts.get)
            present_students.append(name)

        # Draw rectangle and name
        top, right, bottom, left = box
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Save annotated image
    annotated_path = os.path.join(os.path.dirname(file_path), "annotated_" + os.path.basename(file_path))
    cv2.imwrite(annotated_path, image)

    # Prepare attendance list
    all_students = os.listdir(DATASET_DIR)
    absent_students = [s for s in all_students if s not in present_students]

    # Save to CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Student Name", "Status"])
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for student in present_students:
            writer.writerow([today, student, "Present"])
        for student in absent_students:
            writer.writerow([today, student, "Absent"])

    messagebox.showinfo("Success",
                        f"Attendance done!\nAnnotated Image: {annotated_path}\nCSV Updated: {CSV_FILE}")

# Tkinter GUI
root = tk.Tk()
root.title("Classroom Attendance System")
root.geometry("400x150")

label = tk.Label(root, text="Click the button to select a group photo and mark attendance", wraplength=380)
label.pack(pady=20)

btn = tk.Button(root, text="Select Photo & Mark Attendance", command=mark_attendance, width=30, height=2)
btn.pack(pady=10)

root.mainloop()
