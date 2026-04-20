"""
utils.py
========
Shared utilities for the ISL (Indian Sign Language) recording and
processing pipeline.

Used by:
    - record_isl_video.py
    - split_isl_video.py
"""

import cv2
import numpy as np

# ── Updated Defaults (tuned for 1.5-3.0 motion range) ──────────────────────────
STILLNESS_THRESHOLD  = 1.3     # Lowered from 8 based on user feedback
STILLNESS_MIN_FRAMES = 30      # Increased from 12 (approx 1 second at 30fps)


# ── Motion Detection ──────────────────────────────────────────────────────────
def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """
    Return mean pixel difference between two BGR frames (0 = identical).
    Includes a slight blur to filter out camera sensor noise.
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur to reduce noise (prevents 0.9 base-noise from spiking)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Optional: could use a threshold here to ignore tiny pixel flickers
    # _, diff = cv2.threshold(diff, 3, 255, cv2.THRESH_TOZERO)
    
    return float(np.mean(diff))


# ── Camera Probe ──────────────────────────────────────────────────────────────
def open_camera(max_index: int = 4) -> cv2.VideoCapture | None:
    """
    Try camera indices 0..max_index-1.
    Prefers CAP_DSHOW on Windows for faster init.
    Returns the first working VideoCapture, or None.
    """
    # Round 1 — with DirectShow
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  📷 Camera opened at index {idx} (DSHOW)")
                return cap
            cap.release()

    # Round 2 — fallback without DSHOW
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  📷 Camera opened at index {idx}")
                return cap
            cap.release()

    return None
