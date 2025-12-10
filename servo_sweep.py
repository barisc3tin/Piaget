from adafruit_servokit import ServoKit
import time

# ---------- AYARLAR ----------
SERVO_CHANNEL = 15   # hangi kanalı test etmek istiyorsan burayı değiştir
MIN_ANGLE = 85      # önce ÇOK DAR bir aralıktan başlıyoruz
MAX_ANGLE = 95
STEP = 1            # her adımda kaç derece oynasın
DELAY = 0.2         # hareket arası bekleme süresi (saniye)

kit = ServoKit(channels=16)
kit.servo[SERVO_CHANNEL].set_pulse_width_range(500, 2500)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def main():
    print("=== Basit Servo Sweep Testi ===")
    print(f"Kanal: {SERVO_CHANNEL}")
    print(f"Açı aralığı: {MIN_ANGLE}° - {MAX_ANGLE}°")
    print("ÇIKIŞ: Ctrl+C")

    angle = (MIN_ANGLE + MAX_ANGLE) / 2
    angle = clamp(angle, MIN_ANGLE, MAX_ANGLE)
    kit.servo[SERVO_CHANNEL].angle = angle
    time.sleep(1)

    direction = 1

    try:
        while True:
            angle += direction * STEP
            if angle >= MAX_ANGLE:
                angle = MAX_ANGLE
                direction = -1
            elif angle <= MIN_ANGLE:
                angle = MIN_ANGLE
                direction = 1

            kit.servo[SERVO_CHANNEL].angle = angle
            print(f"Açı: {angle:.1f}°")
            time.sleep(DELAY)

    except KeyboardInterrupt:
        print("\nDurduruldu. Son açı:", angle)
        # İstersen sonunda belli bir açıda bırak:
        # kit.servo[SERVO_CHANNEL].angle = (MIN_ANGLE + MAX_ANGLE) / 2

if __name__ == "__main__":
    main()
