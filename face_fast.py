import cv2

# --- AYARLAR ---
CAM_INDEX = 0   # gerekirse 1 veya 2 yaparsın
WIDTH = 320
HEIGHT = 240
DOWNSCALE = 0.5  # algılama için küçültme oranı (0.5 = %50)

CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
# Eğer sende bu klasör değilse, daha önce bulduğumuz yolu yaz:
# CASCADE_PATH = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Yüz cascade yüklenemedi: {CASCADE_PATH}")

cap = cv2.VideoCapture(CAM_INDEX)

# Kamera çözünürlüğünü düşür
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

print("Face detection (fast mode) başlıyor. Kapatmak için 'q'ya bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gri ton ve küçültme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    small_gray = cv2.resize(
        gray,
        (0, 0),
        fx=DOWNSCALE,
        fy=DOWNSCALE
    )

    # Daha hızlı ama hâlâ işe yarar parametreler
    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,    # 1.1 -> daha hızlı, 1.3 -> daha hızlı ama bazen kaçırır
        minNeighbors=4,     # daha düşük = daha fazla/çabuk algılama
        minSize=(40, 40)
    )

    # Küçük görüntüde bulduğumuz yüzleri orijinal boyuta geri ölçekle
    for (x, y, w, h) in faces:
        x = int(x / DOWNSCALE)
        y = int(y / DOWNSCALE)
        w = int(w / DOWNSCALE)
        h = int(h / DOWNSCALE)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Piaget Fast Face Detection", frame)

    # q'ya basınca çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
