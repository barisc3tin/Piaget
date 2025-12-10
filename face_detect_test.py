import cv2
import os

def find_haar_cascade():
    # Muhtemel klasörler
    candidates = [
        "/usr/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
    ]

    for base in candidates:
        path = os.path.join(base, "haarcascade_frontalface_default.xml")
        if os.path.exists(path):
            print("Using cascade:", path)
            return path

    raise RuntimeError("haarcascade_frontalface_default.xml not found in known locations")

cascade_path = find_haar_cascade()

face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)  # gerekirse 1 veya 2

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Piaget Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
