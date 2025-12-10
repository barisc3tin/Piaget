import cv2
import time
from collections import deque
from adafruit_servokit import ServoKit

# ---------- CAMERA SETTINGS ----------
CAM_INDEX = 0
WIDTH = 640
HEIGHT = 480

# Detection için küçültme oranı:
# 1.0 = hiç küçültme yok (daha net, daha yavaş)
# 0.75 = orta yol
# 0.5 = çok hızlı, biraz daha zor algılama
DOWNSCALE = 0.75

# ---------- CASCADE PATH ----------
# Gerekirse burayı kendi sistemindeki gerçek path ile değiştir.
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
# Örn: "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Yüz cascade yüklenemedi: {CASCADE_PATH}")

# ---------- SERVO SETTINGS ----------
kit = ServoKit(channels=16)

NECK_YAW_CH = 1    # sağ-sol
NECK_TILT_CH = 0   # yukarı-aşağı

# YAW: 90 = sol, 135 = orta, 180 = sağ
YAW_MIN = 90
YAW_MAX = 180
YAW_CENTER = 135

# TILT: kabaca güvenli aralık (mekaniğe göre ayarlanabilir)
TILT_MIN = 60
TILT_MAX = 120
TILT_CENTER = 90

for ch in [NECK_YAW_CH, NECK_TILT_CH]:
    kit.servo[ch].set_pulse_width_range(500, 2500)

yaw_angle = YAW_CENTER
tilt_angle = TILT_CENTER

kit.servo[NECK_YAW_CH].angle = yaw_angle
kit.servo[NECK_TILT_CH].angle = tilt_angle
time.sleep(1)

# ---------- CONTROL PARAMS ----------
# Daha yumuşak, daha az titreyen hareket için düşük gain + küçük step
YAW_GAIN = 0.025
TILT_GAIN = 0.025
MAX_STEP = 2           # bir güncellemede max derece
DEAD_ZONE_X = 35       # merkez çevresinde tepki vermeme bölgesi
DEAD_ZONE_Y = 25

# Yüz kaybolunca merkeze dönme
NO_FACE_TIMEOUT = 1.5  # saniye
last_seen_time = time.time()

# Yüz merkezini yumuşatmak için son N değerin ortalaması
FACE_SMOOTH_WINDOW = 4
face_centers = deque(maxlen=FACE_SMOOTH_WINDOW)

# Servoları her framede değil, her N framede güncelle
frame_count = 0
UPDATE_EVERY = 2

# ---------- CAMERA ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Kamera açılamadı.")
    raise SystemExit

print("Piaget head tracking (smooth + auto-center) başlıyor. Çıkmak için 'q'.")

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection için küçült
    small_gray = cv2.resize(
        gray,
        (0, 0),
        fx=DOWNSCALE,
        fy=DOWNSCALE
    )

    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    target_center = None

    if len(faces) > 0:
        # En büyük yüzü seç (kameraya en yakın)
        largest = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = largest

        # Küçük görüntüden orijinale ölçekle
        x = int(x / DOWNSCALE)
        y = int(y / DOWNSCALE)
        w = int(w / DOWNSCALE)
        h = int(h / DOWNSCALE)

        cx = x + w // 2
        cy = y + h // 2

        # Yumuşatma buffer’ına ekle
        face_centers.append((cx, cy))

        # Ortalama al → yumuşatılmış hedef
        avg_x = int(sum(p[0] for p in face_centers) / len(face_centers))
        avg_y = int(sum(p[1] for p in face_centers) / len(face_centers))
        target_center = (avg_x, avg_y)

        # Görselleştirme
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (avg_x, avg_y), 4, (0, 255, 0), -1)

        last_seen_time = time.time()
    else:
        # Yüz yokken önceki merkezleri sıfırla
        face_centers.clear()

    # Ekran merkezi
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    now = time.time()

    # ----- SERVO CONTROL -----
    # Servoları her framede değil, her UPDATE_EVERY framede güncelle
    if frame_count % UPDATE_EVERY == 0:
        if target_center is not None:
            face_x, face_y = target_center

            error_x = face_x - center_x   # sağ +, sol -
            error_y = face_y - center_y   # aşağı +, yukarı -

            # YAW (sağ-sol)
            if abs(error_x) > DEAD_ZONE_X:
                delta_yaw = clamp(error_x * YAW_GAIN, -MAX_STEP, MAX_STEP)
                yaw_angle = clamp(yaw_angle + delta_yaw, YAW_MIN, YAW_MAX)
                kit.servo[NECK_YAW_CH].angle = yaw_angle

            # TILT (yukarı-aşağı)
            if abs(error_y) > DEAD_ZONE_Y:
                delta_tilt = clamp(error_y * TILT_GAIN, -MAX_STEP, MAX_STEP)
                tilt_angle = clamp(tilt_angle + delta_tilt, TILT_MIN, TILT_MAX)
                kit.servo[NECK_TILT_CH].angle = tilt_angle

        else:
            # Yüz yok → belli süre sonra merkeze yumuşak dönüş
            if now - last_seen_time > NO_FACE_TIMEOUT:
                if abs(yaw_angle - YAW_CENTER) > 0.5:
                    step = clamp(YAW_CENTER - yaw_angle, -MAX_STEP, MAX_STEP)
                    yaw_angle = clamp(yaw_angle + step, YAW_MIN, YAW_MAX)
                    kit.servo[NECK_YAW_CH].angle = yaw_angle

                if abs(tilt_angle - TILT_CENTER) > 0.5:
                    step = clamp(TILT_CENTER - tilt_angle, -MAX_STEP, MAX_STEP)
                    tilt_angle = clamp(tilt_angle + step, TILT_MIN, TILT_MAX)
                    kit.servo[NECK_TILT_CH].angle = tilt_angle

    # Görüntüyü büyüt, FPS’e çok dokunmadan daha büyük göster
    display = cv2.resize(frame, None, fx=1.5, fy=1.5)
    cv2.imshow("Piaget Head Tracking (Smooth + Auto-center)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
