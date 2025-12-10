from adafruit_servokit import ServoKit
import time

# ---------- AYARLAR ----------
JAW_CH = 4

# Güvenli deneme aralığı (mekaniğe göre istersen daralt)
MIN_ANGLE = 60
MAX_ANGLE = 120

# Varsayılan başlangıç açısı (emin değilsen 90 yaz)
DEFAULT_START_ANGLE = 90

kit = ServoKit(channels=16)
kit.servo[JAW_CH].set_pulse_width_range(500, 2500)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def main():
    print("=== Piaget Jaw Servo Test (CH4) ===")
    print(f"Güvenli açı aralığı: {MIN_ANGLE}° - {MAX_ANGLE}°")
    print("UYARI: Mekanizma zorlanırsa anında güç kablosunu çekmeyi hazırda tut!")
    print()

    # Başlangıç açısını kullanıcıdan al
    try:
        txt = input(f"Başlangıç açısı? (Enter -> {DEFAULT_START_ANGLE}): ").strip()
        if txt == "":
            angle = DEFAULT_START_ANGLE
        else:
            angle = float(txt)
    except Exception:
        angle = DEFAULT_START_ANGLE

    angle = clamp(angle, MIN_ANGLE, MAX_ANGLE)
    print(f"Başlangıç açısı {angle}° olarak ayarlandı.")
    kit.servo[JAW_CH].angle = angle
    time.sleep(0.5)

    print()
    print("Kontroller:")
    print("  a  -> -1° (küçük geri)")
    print("  d  -> +1° (küçük ileri)")
    print("  z  -> -5° (büyük geri)")
    print("  c  -> +5° (büyük ileri)")
    print("  s  -> mevcut açıyı yazdır")
    print("  q  -> çıkış")
    print()

    try:
        while True:
            cmd = input("Komut (a/d/z/c/s/q): ").strip().lower()
            if cmd == "q":
                print("Çıkılıyor. Servo şu açıda bırakılıyor:", angle)
                break
            elif cmd == "s":
                print(f"Mevcut açı: {angle}°")
            elif cmd == "a":
                angle -= 1
            elif cmd == "d":
                angle += 1
            elif cmd == "z":
                angle -= 5
            elif cmd == "c":
                angle += 5
            else:
                print("Geçersiz komut.")
                continue

            old_angle = angle
            angle = clamp(angle, MIN_ANGLE, MAX_ANGLE)
            if angle != old_angle:
                print(f"Sınır nedeniyle açı {angle}°'ye kısıtlandı.")

            print(f"-> Yeni açı: {angle}°")
            kit.servo[JAW_CH].angle = angle
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nKlavye kesmesi. Son açı:", angle)


if __name__ == "__main__":
    main()
