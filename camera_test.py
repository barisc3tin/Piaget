# camera_test.py
import cv2

cap = cv2.VideoCapture(0)  # gerekirse 1, 2 yaparız

if not cap.isOpened():
    print("Kameraya erişilemiyor.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame okunamadı.")
        break

    cv2.imshow("Piaget Camera", frame)

    # q'ya basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
