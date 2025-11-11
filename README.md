cat > README.md <<'EOF'
# Classroom Attendance ML

Simple face-recognition based attendance system.
- encode faces from `dataset/` into `encodings.pickle`
- detect faces in a classroom/group photo and mark attendance
- GUI (Tkinter) to select photo and export attendance CSV

**Note:** Do NOT push `dataset/`, `encodings.pickle`, or photos (they are in .gitignore).
EOF
