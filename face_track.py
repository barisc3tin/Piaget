import cv2
import time
from adafruit_servokit import ServoKit

# ---------- CAMERA SETTINGS ----------
CAM_INDEX = 0
WIDTH = 640
HEIGHT = 480
DOWNSCALE = 0.75  # detection için küçültme oranı

# ---------- CASCADE PATH ----------
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
# Gerekirse burayı kendi yolunla değiştir

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

# TILT: yaklaşık güvenli aralık
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
YAW_GAIN = 0.04
TILT_GAIN = 0.04
MAX_STEP = 4
DEAD_ZONE_X = 20
DEAD_ZONE_Y = 15

# Yüz kaybolduğunda merkeze dönme davranışı
NO_FACE_TIMEOUT = 1.5  # saniye (1.5 sn yüz yoksa merkeze dön)
last_seen_time = time.time()

# ---------- CAMERA ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

print("Piaget head tracking (auto-center) başlıyor. Çıkmak için 'q'.")

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)

    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    target_center = None

    if len(faces) > 0:
        # En büyük yüzü seç
        largest = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = largest

        x = int(x / DOWNSCALE)
        y = int(y / DOWNSCALE)
        w = int(w / DOWNSCALE)
        h = int(h / DOWNSCALE)

        cx = x + w // 2
        cy = y + h // 2
        target_center = (cx, cy)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Yüz gördük → zamanı güncelle
        last_seen_time = time.time()

    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    now = time.time()

    # ----- SERVO CONTROL -----
    if target_center is not None:
        # YÜZ VAR → takip et
        face_x, face_y = target_center

        error_x = face_x - center_x   # sağ +, sol -
        error_y = face_y - center_y   # aşağı +, yukarı -

        # YAW
        if abs(error_x) > DEAD_ZONE_X:
            delta_yaw = clamp(error_x * YAW_GAIN, -MAX_STEP, MAX_STEP)
            yaw_angle = clamp(yaw_angle + delta_yaw, YAW_MIN, YAW_MAX)
            kit.servo[NECK_YAW_CH].angle = yaw_angle

        # TILT
        if abs(error_y) > DEAD_ZONE_Y:
            delta_tilt = clamp(error_y * TILT_GAIN, -MAX_STEP, MAX_STEP)
            tilt_angle = clamp(tilt_angle + delta_tilt, TILT_MIN, TILT_MAX)
            kit.servo[NECK_TILT_CH].angle = tilt_angle

    else:
        # YÜZ YOK → belli süre geçtiyse merkeze yavaşça dön
        if now - last_seen_time > NO_FACE_TIMEOUT:
            # Yaw'ı merkeze yaklaştır
            if abs(yaw_angle - YAW_CENTER) > 0.5:
                step = clamp(YAW_CENTER - yaw_angle, -MAX_STEP, MAX_STEP)
                yaw_angle = clamp(yaw_angle + step, YAW_MIN, YAW_MAX)
                kit.servo[NECK_YAW_CH].angle = yaw_angle

            # Tilt'i merkeze yaklaştır
            if abs(tilt_angle - TILT_CENTER) > 0.5:
                step = clamp(TILT_CENTER - tilt_angle, -MAX_STEP, MAX_STEP)
                tilt_angle = clamp(tilt_angle + step, TILT_MIN, TILT_MAX)
                kit.servo[NECK_TILT_CH].angle = tilt_angle

    # Görüntüyü büyüt
    display = cv2.resize(frame, None, fx=1.5, fy=1.5)
    cv2.imshow("Piaget Head Tracking (Auto-center)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
