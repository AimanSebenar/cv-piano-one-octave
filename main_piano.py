import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import numpy as np
import sounddevice as sd
import threading
import time
import urllib.request
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved to: {MODEL_PATH}")
else:
    print(f"Model found: {MODEL_PATH}")

KEYS = [["C4",  60, "white"],
    ["C#4", 61, "black"],
    ["D4",  62, "white"],
    ["D#4", 63, "black"],
    ["E4",  64, "white"],
    ["F4",  65, "white"],
    ["F#4", 66, "black"],
    ["G4",  67, "white"],
    ["G#4", 68, "black"],
    ["A4",  69, "white"],
    ["A#4", 70, "black"],
    ["B4",  71, "white"],
    ["C5",  72, "white"]]

FRAME_W  =1280
FRAME_H = 720
PIANO_H = 160
PIANO_Y = FRAME_H - PIANO_H

WHITE_KEYS = [k for k in KEYS if k[2] == 'white']
KEY_W = FRAME_W // len(WHITE_KEYS)

white_idx = 0
for key in KEYS:
    if key[2] == 'white':
        key += [white_idx * KEY_W, PIANO_Y, KEY_W - 1, PIANO_H]
        white_idx += 1

    else:
        black_w = int(KEY_W * 0.55)
        black_h = int(PIANO_H * 0.60)
        whites_prev = sum(1 for w in WHITE_KEYS if w[1] < key[1])
        key += [whites_prev * KEY_W - black_w // 2, PIANO_Y, black_w, black_h]

SAMPLE_RATE = 44100

def play_note(midi):
    freq = 440.0 * (2.0 **((midi - 69) / 12.0)) #conv MIDI to Hz

    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    wave = np.sin(2* np.pi * freq * t).astype(np.float32)

    wave *= np.linspace(1.0, 0.0, SAMPLE_RATE)

    threading.Thread(target=sd.play, args=(wave, SAMPLE_RATE), daemon=True).start()

options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

prev_key = None
last_played = {}
COOLDOWN = 0.5

print('Point index finger at key. Press Q to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, ts_ms)

    tip_key = None

    if result.hand_landmarks:

        lm = result.hand_landmarks[0]

        tip_x = int(lm[8].x * FRAME_W)
        tip_y = int(lm[8].y * FRAME_H)

        cv2.circle(frame, (tip_x, tip_y), 12, (0, 210, 255), -1)
        cv2.putText(frame, 'tip', (tip_x + 14, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1)

        for key in [k for k in KEYS if k[2] == 'black'] + [k for k in KEYS if k[2] == 'white']:
            x, y, w, h = key[3], key[4], key[5], key[6]

            if x <= tip_x <= x+w and y <= tip_y <= y+h:
                tip_key = key
                break

        if tip_key:
            name = tip_key[0]
            now=time.time()
            new_key = (prev_key is None or prev_key != name)
            cooled = (now - last_played.get(name, 0) >= COOLDOWN)

            if new_key and cooled:
                play_note(tip_key[1])
                last_played[name] = now
                print(f'{name} note')

        prev_key = tip_key[0] if tip_key else None

    for key in [k for k in KEYS if k[2] == 'white'] + [k for k in KEYS if k[2] == 'black']:
        x, y, w, h = key[3], key[4], key[5], key[6]
        active = tip_key is not None and tip_key[0] == key[0]

        if key[2] == 'white':
            colour = (60, 185, 255) if active else (230, 225, 210)
        else:
            colour = (30, 140, 220) if active else (30, 30, 30)

        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, -1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (80, 70, 60), 1)

        if key[2] == 'white':
            label = key[0].rstrip("0123456789")
            cv2.putText(frame, label, (x + w // 2 - 7, y + h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            
    if tip_key:
        cv2.putText(frame, tip_key[0], (FRAME_W // 2 - 30, PIANO_Y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 200, 255), 3)

    cv2.putText(frame, "Press Q to quit", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imshow("CV Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()