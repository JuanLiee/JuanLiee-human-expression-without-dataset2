import cv2
import numpy as np
import tensorflow as tf

# =====================
# LOAD MODEL
# =====================
MODEL_PATH = "emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 48

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# =====================
# LOAD FACE DETECTOR
# =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================
# OPEN WEBCAM
# =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam can't be opened")
    exit()

print("press q to quit")

# =====================
# REAL-TIME LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)
        emotion = class_names[np.argmax(prediction)]

        # DRAW BOX & TEXT
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detection - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================
# CLEAN UP
# =====================
cap.release()
cv2.destroyAllWindows()
