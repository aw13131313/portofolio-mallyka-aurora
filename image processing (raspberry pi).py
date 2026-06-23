import cv2
import os
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Setup GPIO
BUTTON_PIN = 17  # Ganti jika pakai pin lain

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Load cascade file
face_detector = cv2.CascadeClassifier(
    "/home/pi/Face/haarcascade_frontalface_default.xml"
)

cv2.startWindowThread()

# Setup camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={
            "format": "XRGB8888",
            "size": (1280, 720)
        }
    )
)
picam2.start()

# Create a directory to store detected faces
output_directory = "detected_faces"
os.makedirs(output_directory, exist_ok=True)

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("Tombol ditekan, keluar dari program...")
            break

        im = picam2.capture_array()

        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(grey, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Generate a unique filename using timestamp
            timestamp = int(time.time())
            filename = os.path.join(
                output_directory,
                f"face_{timestamp}.jpg"
            )

            # Save only the detected face portion
            cv2.imwrite(filename, im[y:y + h, x:x + w])

        cv2.imshow("Camera", im)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("KeyboardInterrupt - program dihentikan.")

finally:
    cv2.destroyAllWindows()
    GPIO.cleanup()