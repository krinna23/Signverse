"""
camera_test.py
==============
Real-time hand gesture recognition using webcam + MediaPipe HandLandmarker.

Detects up to 2 hands (left & right) simultaneously.
Displays the predicted gesture character on screen.

Controls
--------
  1  →  Switch to MLP model
  2  →  Switch to Random Forest model
  3  →  Switch to KNN model  (default)
  q  →  Quit
"""

import os
import sys
import pickle
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH         = 'hand_landmarker.task'
FEATURES_PER_HAND  = 63      # 21 landmarks * (x, y, z)
ZERO_HAND          = [0.0] * FEATURES_PER_HAND
CONFIDENCE         = 0.5

MODEL_FILES = {
    '1': ('MLP',           'model_mlp.p'),
    '2': ('Random Forest', 'model_rf.p'),
    '3': ('KNN',           'model_knn.p'),
}
DEFAULT_MODEL_KEY = '3'

FONT       = cv2.FONT_HERSHEY_SIMPLEX
COLOR_PRED = (50, 255, 50)
COLOR_INFO = (255, 220, 0)
COLOR_LAND = (0, 255, 180)
COLOR_CONN = (180, 0, 255)
COLOR_WARN = (0, 80, 255)
COLOR_BOX  = (30, 30, 30)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Validate MediaPipe model file ──────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] {MODEL_PATH} not found.")
    sys.exit(1)

# ── Load trained classifiers + label encoders ──────────────────────────────────
models = {}   # key -> (display_name, classifier, label_encoder)
for key, (name, path) in MODEL_FILES.items():
    if os.path.exists(path):
        try:
            pkg = pickle.load(open(path, 'rb'))
            clf = pkg['model']
            le  = pkg.get('label_encoder', None)   # may be None for old pickles
            models[key] = (name, clf, le)
            print(f"  Loaded [{key}] {name}  ({path})")
        except Exception as e:
            print(f"  [WARN] Could not load {path}: {e}")
    else:
        print(f"  [WARN] {path} not found – {name} unavailable.")

if not models:
    print("[ERROR] No classifier models found. Run train.py first.")
    sys.exit(1)

current_key = DEFAULT_MODEL_KEY if DEFAULT_MODEL_KEY in models else list(models.keys())[0]
print(f"\n  Default model: {models[current_key][0]} (key {current_key})\n")

# ── MediaPipe HandLandmarker ───────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=CONFIDENCE,
)
detector = vision.HandLandmarker.create_from_options(options)

# ── Open webcam (probe indices 0-3 until one works) ───────────────────────────
def open_camera(max_index: int = 4) -> cv2.VideoCapture:
    """Try camera indices 0..max_index-1; return first working capture."""
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW faster on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  Camera opened at index {idx}")
                return cap
            cap.release()
    # Fallback without CAP_DSHOW
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  Camera opened at index {idx} (no DSHOW)")
                return cap
            cap.release()
    return None

cap = open_camera()
if cap is None:
    print("[ERROR] Could not open any webcam (tried indices 0-3).")
    detector.close()
    sys.exit(1)

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_hand_features(hand_landmarks) -> list:
    x_vals = [lm.x for lm in hand_landmarks]
    y_vals = [lm.y for lm in hand_landmarks]
    min_x, min_y = min(x_vals), min(y_vals)
    features = []
    for lm in hand_landmarks:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)
        features.append(lm.z)
    return features


def draw_hand_skeleton(frame, hand_landmarks, H, W):
    pts = [(int(lm.x * W), int(lm.y * H)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_CONN, 1, cv2.LINE_AA)
    for px, py in pts:
        cv2.circle(frame, (px, py), 4, COLOR_LAND, -1, cv2.LINE_AA)


def put_text_with_bg(frame, text, origin, font_scale=1.0,
                     color=COLOR_PRED, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = origin
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-4, y-th-6), (x+tw+4, y+baseline+2), COLOR_BOX, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def build_status_bar(frame, H, W):
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - bar_h), (W, H), COLOR_BOX, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    model_name = models[current_key][0] if current_key in models else "None"
    text = f"  Active: {model_name}   |   [1] MLP  [2] RF  [3] KNN  [q] Quit"
    cv2.putText(frame, text, (6, H - 10), FONT, 0.5, COLOR_INFO, 1, cv2.LINE_AA)


def decode_prediction(raw_pred, le):
    """Convert model output to a human-readable label string."""
    if le is not None:
        # Model trained on integer-encoded labels → decode back to class name
        return str(le.inverse_transform([int(raw_pred)])[0])
    # Old model without encoder → assume output is already the label
    return str(raw_pred)

# ── Webcam loop ────────────────────────────────────────────────────────────────
print("Camera started. Press 1/2/3 to switch models, 'q' to quit.\n")
consecutive_fail = 0
MAX_CONSECUTIVE_FAIL = 30   # exit if 30 frames in a row fail

while True:
    ret, frame = cap.read()

    if not ret:
        consecutive_fail += 1
        if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
            print(f"[ERROR] {MAX_CONSECUTIVE_FAIL} consecutive read failures. Exiting.")
            break
        continue

    consecutive_fail = 0   # reset on success
    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)   # mirror for natural feel

    img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)

    prediction_text = ""

    if result.hand_landmarks:
        hand_data = {}

        for idx, handedness_list in enumerate(result.handedness):
            side = handedness_list[0].category_name   # 'Left' or 'Right'
            lms  = result.hand_landmarks[idx]

            draw_hand_skeleton(frame, lms, H, W)

            # Label above wrist
            wrist = lms[0]
            wx, wy = int(wrist.x * W), max(int(wrist.y * H) - 15, 15)
            cv2.putText(frame, side, (wx, wy), FONT, 0.55, COLOR_INFO, 1, cv2.LINE_AA)

            hand_data[side] = extract_hand_features(lms)

        left_feats  = hand_data.get('Left',  ZERO_HAND)
        right_feats = hand_data.get('Right', ZERO_HAND)
        if len(left_feats)  != FEATURES_PER_HAND: left_feats  = ZERO_HAND
        if len(right_feats) != FEATURES_PER_HAND: right_feats = ZERO_HAND

        feature_vector = np.array(left_feats + right_feats,
                                  dtype=np.float64).reshape(1, -1)

        if current_key in models:
            model_name, clf, le = models[current_key]
            raw_pred = clf.predict(feature_vector)[0]
            prediction_text = decode_prediction(raw_pred, le)

    # ── Overlay ───────────────────────────────────────────────────────────────
    if prediction_text:
        put_text_with_bg(frame, prediction_text,
                         (W // 2 - 40, 90),
                         font_scale=2.8, color=COLOR_PRED, thickness=4)
    else:
        put_text_with_bg(frame, "No hand detected",
                         (10, 50),
                         font_scale=0.75, color=COLOR_WARN, thickness=2)

    build_status_bar(frame, H, W)
    cv2.imshow('Hand Gesture Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting.")
        break
    elif chr(key) in models:
        current_key = chr(key)
        print(f"  Switched to: {models[current_key][0]}")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
detector.close()
