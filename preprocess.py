"""
preprocess.py
=============
Extracts hand landmarks from images in data/new_data using MediaPipe HandLandmarker.
Produces a 126-feature vector per sample:
    - 63 values for Left hand  (21 landmarks * x, y, z), zero-padded if absent
    - 63 values for Right hand (21 landmarks * x, y, z), zero-padded if absent
Landmarks are normalized relative to each hand's bounding box min-x / min-y.

Output: data.pickle  ->  {'data': List[List[float]], 'labels': List[str]}
"""

import os
import pickle
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH         = 'hand_landmarker.task'
DATA_DIR           = './data/new_data'
OUTPUT_FILE        = 'data.pickle'
NUM_HANDS          = 2
FEATURES_PER_HAND  = 63      # 21 landmarks * (x, y, z)
TOTAL_FEATURES     = 126
ZERO_HAND          = [0.0] * FEATURES_PER_HAND
SUPPORTED_EXT      = {'.jpg', '.jpeg', '.png'}

# ── Validate paths ─────────────────────────────────────────────────────────────
for path, label in [(MODEL_PATH, 'MediaPipe model'), (DATA_DIR, 'Data directory')]:
    if not os.path.exists(path):
        print(f"[ERROR] {label} not found: {path}")
        sys.exit(1)

# ── MediaPipe detector ─────────────────────────────────────────────────────────
# Only pass min_hand_detection_confidence — other tracking params cause crashes
# on some mediapipe 0.10.x builds.
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=0.3,
)
detector = vision.HandLandmarker.create_from_options(options)

# ── Helpers ────────────────────────────────────────────────────────────────────
def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXT


def extract_hand_features(hand_landmarks) -> list:
    """Return a 63-element normalized landmark list for one hand."""
    x_vals = [lm.x for lm in hand_landmarks]
    y_vals = [lm.y for lm in hand_landmarks]
    min_x, min_y = min(x_vals), min(y_vals)
    features = []
    for lm in hand_landmarks:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)
        features.append(lm.z)
    return features

# ── Main loop ──────────────────────────────────────────────────────────────────
data:   list = []
labels: list = []

# Per-class stats: label -> {'total': int, 'detected': int}
class_stats: dict = {}

skipped_no_hand  = 0
skipped_bad_read = 0

# Sort folders so order is deterministic (0,1,...,10,...,A,B,...,Z)
folders = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)

print(f"Found {len(folders)} label folders in {DATA_DIR}\n")

for folder in folders:
    label    = folder.split('-')[0]   # "A-samples" -> "A", "10-samples" -> "10"
    dir_path = os.path.join(DATA_DIR, folder)
    images   = [f for f in os.listdir(dir_path) if is_image(f)]
    total    = len(images)

    class_stats[label] = {'total': total, 'detected': 0}
    print(f"  Processing [{label:>3}]  {total} images ...", end='', flush=True)

    for img_file in images:
        img_bgr = cv2.imread(os.path.join(dir_path, img_file))

        if img_bgr is None:
            skipped_bad_read += 1
            continue

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_image)

        if not result.hand_landmarks:
            skipped_no_hand += 1
            continue

        hand_data = {}
        for idx, handedness_list in enumerate(result.handedness):
            side     = handedness_list[0].category_name   # 'Left' or 'Right'
            features = extract_hand_features(result.hand_landmarks[idx])
            hand_data[side] = features

        left_feats  = hand_data.get('Left',  ZERO_HAND)
        right_feats = hand_data.get('Right', ZERO_HAND)

        if len(left_feats)  != FEATURES_PER_HAND: left_feats  = ZERO_HAND
        if len(right_feats) != FEATURES_PER_HAND: right_feats = ZERO_HAND

        data.append(left_feats + right_feats)
        labels.append(label)
        class_stats[label]['detected'] += 1

    detected = class_stats[label]['detected']
    rate     = (detected / total * 100) if total > 0 else 0.0
    print(f"  {detected}/{total} detected ({rate:.0f}%)")

detector.close()

# ── Save ───────────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# ── Per-class summary table ────────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"  {'Class':<8} {'Detected':>10} {'Total':>8}  {'Rate':>7}")
print(f"{'─'*52}")

for lbl in sorted(class_stats.keys(), key=lambda x: (len(x), x)):
    stats    = class_stats[lbl]
    detected = stats['detected']
    total    = stats['total']
    rate     = (detected / total * 100) if total > 0 else 0.0
    flag     = "  ⚠ low" if rate < 50 else ""
    print(f"  {lbl:<8} {detected:>10} {total:>8}  {rate:>6.1f}%{flag}")

print(f"{'─'*52}")
print(f"\n  Total samples saved  : {len(data)}")
print(f"  Skipped (no hand)    : {skipped_no_hand}")
print(f"  Skipped (bad image)  : {skipped_bad_read}")
print(f"  Output               : {OUTPUT_FILE}")
print(f"{'─'*52}")
