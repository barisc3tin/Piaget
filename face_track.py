import cv2
import time
from collections import deque
from adafruit_servokit import ServoKit
import subprocess  # <-- audio için

# ---------- CAMERA SETTINGS ----------
CAM_INDEX = 0
WIDTH = 320
HEIGHT = 240

# Detection için küçültme oranı
DOWNSCALE = 1

# ---------- AUDIO SETTINGS ----------
AUDIO_FILES = ["piaget_0.wav", "piaget_1.wav", "piaget_2.wav"]
GREET_COOLDOWN = 5.0  # aynı kişiyi sürekli görünce spam yapmasın (saniye)
last_greet_time = 0.0
greet_index = 0

def play_greet(index):
    """Belirtilen indexteki wav dosyasını çal (non-blocking)."""
    if index < 0 or index >= len(AUDIO_FILES):
        return
    filename = AUDIO_FILES[index]
    try:
        # aplay ile çal, stdout/stderr'i sustur
        subprocess.Popen(
            ["aplay", filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Playing: {filename}")
    except Exception as e:
        print("Audio play error:", e)

# ---------- CASCADE PATH ----------
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
# Gerekirse kendi yolunla değiştir:
# CASCADE_PATH = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"

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

# TILT: ayarlanabilir güvenli aralık
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

# ---------- CONTROL PARAMS (RULE-BASED) ----------

# Hata bölgeleri:
DEAD_ZONE_SMALL_X = 35   # merkez civarı, hiç hareket yok
DEAD_ZONE_LARGE_X = 120  # çok uzakta → büyük adım

DEAD_ZONE_SMALL_Y = 18
DEAD_ZONE_LARGE_Y = 60

# Servo adımları (derece)
YAW_STEP_SMALL = 2
YAW_STEP_LARGE = 5

TILT_STEP_SMALL = 2
TILT_STEP_LARGE = 4

# Yüz kaybolunca merkeze dönme ve scan
NO_FACE_TIMEOUT = 1.5  # yüz yoksa, bu kadar saniye sonra "return to neutral"
last_seen_time = time.time()

# Yüz merkezini yumuşatma
FACE_SMOOTH_WINDOW = 4
face_centers = deque(maxlen=FACE_SMOOTH_WINDOW)

# Servoları her framede değil, her N framede güncelle
frame_count = 0
UPDATE_EVERY = 1

# Scan davranışı
SCANNING = False
SCAN_STEP = 1          # scan sırasında yaw adımı (derece)
scan_direction = 1     # +1 sağa, -1 sola

# ---------- HELPERS ----------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

def step_towards(current, target, max_step):
    """current değerini target'a doğru en fazla max_step kadar yaklaştır."""
    if abs(target - current) <= max_step:
        return target
    return current + max_step if target > current else current - max_step

# ---------- CAMERA ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Kamera açılamadı.")
    raise SystemExit

print("Piaget head tracking (track + auto-center + scan + audio) başlıyor. Çıkmak için 'q'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection için küçült
    small_gray = cv2.resize(gray, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)

    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40)
    )

    target_center = None

    if len(faces) > 0:
        # FACE FOUND → tracking moduna dön
        SCANNING = False

        # En büyük yüzü seç
        largest = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = largest

        x = int(x / DOWNSCALE)
        y = int(y / DOWNSCALE)
        w = int(w / DOWNSCALE)
        h = int(h / DOWNSCALE)

        cx = x + w // 2
        cy = y + h // 2

        # Yumuşatma buffer’ına ekle
        face_centers.append((cx, cy))

        avg_x = int(sum(p[0] for p in face_centers) / len(face_centers))
        avg_y = int(sum(p[1] for p in face_centers) / len(face_centers))
        target_center = (avg_x, avg_y)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (avg_x, avg_y), 4, (0, 255, 0), -1)

        # Yüz gördük → zamanı güncelle
        last_seen_time = now

        # 🔊 Ses çalma mantığı:
        # Yüz daha önce yoktu, şimdi var VE en son konuşmasından 5 sn geçtiyse bir sonraki sesi çal
        if now - last_greet_time > GREET_COOLDOWN:
            play_greet(greet_index)
            last_greet_time = now
            greet_index = (greet_index + 1) % len(AUDIO_FILES)

    else:
        # Yüz yok → buffer temizle
        face_centers.clear()

    # Ekran merkezi
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

    # ----- SERVO CONTROL -----
    if frame_count % UPDATE_EVERY == 0:
        if target_center is not None:
            # ========== TRACKING MODE ==========
            face_x, face_y = target_center

            error_x = face_x - center_x   # sağ +, sol -
            error_y = face_y - center_y   # aşağı +, yukarı -

            # YAW (sağ-sol) rule-based
            abs_ex = abs(error_x)
            if abs_ex > DEAD_ZONE_SMALL_X:
                if abs_ex > DEAD_ZONE_LARGE_X:
                    step = YAW_STEP_LARGE
                else:
                    step = YAW_STEP_SMALL

                direction = 1 if error_x > 0 else -1
                yaw_angle = clamp(
                    yaw_angle + direction * step,
                    YAW_MIN,
                    YAW_MAX
                )
                kit.servo[NECK_YAW_CH].angle = yaw_angle

            # TILT (yukarı-aşağı) rule-based
            abs_ey = abs(error_y)
            if abs_ey > DEAD_ZONE_SMALL_Y:
                if abs_ey > DEAD_ZONE_LARGE_Y:
                    step = TILT_STEP_LARGE
                else:
                    step = TILT_STEP_SMALL

                direction = 1 if error_y > 0 else -1
                tilt_angle = clamp(
                    tilt_angle + direction * step,
                    TILT_MIN,
                    TILT_MAX
                )
                kit.servo[NECK_TILT_CH].angle = tilt_angle

        else:
            # ========== NO FACE ==========
            time_since_seen = now - last_seen_time

            if time_since_seen <= NO_FACE_TIMEOUT:
                # RETURN PHASE: önce nötre yumuşak dön
                SCANNING = False  # öncelik nötre dönmekte
                if abs(yaw_angle - YAW_CENTER) > 0.5:
                    yaw_angle = step_towards(yaw_angle, YAW_CENTER, YAW_STEP_SMALL)
                    yaw_angle = clamp(yaw_angle, YAW_MIN, YAW_MAX)
                    kit.servo[NECK_YAW_CH].angle = yaw_angle

                if abs(tilt_angle - TILT_CENTER) > 0.5:
                    tilt_angle = step_towards(tilt_angle, TILT_CENTER, TILT_STEP_SMALL)
                    tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                    kit.servo[NECK_TILT_CH].angle = tilt_angle

            else:
                # SCAN MODE: nötre yeterince yakınsak taramaya başla
                if abs(yaw_angle - YAW_CENTER) <= 1.0 and abs(tilt_angle - TILT_CENTER) <= 1.0:
                    SCANNING = True

                if SCANNING:
                    # SCAN: kafa sağ-sol yavaşça tarıyor
                    # Tilt'i merkezde tut
                    if abs(tilt_angle - TILT_CENTER) > 0.5:
                        tilt_angle = step_towards(tilt_angle, TILT_CENTER, TILT_STEP_SMALL)
                        tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                        kit.servo[NECK_TILT_CH].angle = tilt_angle

                    # Yaw'ı tarama için hareket ettir
                    yaw_angle += scan_direction * SCAN_STEP

                    # Sınırları kontrol et, kenara gelince yön değiştir
                    if yaw_angle >= YAW_MAX:
                        yaw_angle = YAW_MAX
                        scan_direction = -1
                    elif yaw_angle <= YAW_MIN:
                        yaw_angle = YAW_MIN
                        scan_direction = 1

                    kit.servo[NECK_YAW_CH].angle = yaw_angle
                else:
                    # Hâlâ nötre yaklaşma aşamasında
                    if abs(yaw_angle - YAW_CENTER) > 0.5:
                        yaw_angle = step_towards(yaw_angle, YAW_CENTER, YAW_STEP_SMALL)
                        yaw_angle = clamp(yaw_angle, YAW_MIN, YAW_MAX)
                        kit.servo[NECK_YAW_CH].angle = yaw_angle

                    if abs(tilt_angle - TILT_CENTER) > 0.5:
                        tilt_angle = step_towards(tilt_angle, TILT_CENTER, TILT_STEP_SMALL)
                        tilt_angle = clamp(tilt_angle, TILT_MIN, TILT_MAX)
                        kit.servo[NECK_TILT_CH].angle = tilt_angle

    # Görüntüyü biraz büyüt
    display = cv2.resize(frame, None, fx=1.5, fy=1.5)
    cv2.imshow("Piaget Head Tracking (Track + Scan + Audio)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
