"""
Hand-Tracking Mouse Controller
================================
Controls your mouse cursor using hand gestures detected via webcam.

Uses the CURRENT MediaPipe Tasks API (mp.tasks.vision.HandLandmarker),
NOT the legacy mp.solutions.hands API.

Requirements:
    pip install opencv-python mediapipe pynput

You also need the hand_landmarker.task model bundle. Download it once:

    python hand_mouse_control.py --download

Or manually:
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )

Gestures:
    - Index finger up only       → move cursor
    - Pinch (index + thumb)      → left click
    - Index + Middle fingers up  → scroll (move hand up/down)
    - Index + Middle + Ring up   → right click
"""

import sys
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# ── Current MediaPipe Tasks API imports ───────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ── Mouse backend ─────────────────────────────────────────────────────────────
# pynput: lower latency (recommended)
# pyautogui: easier API but has a forced 0.1s delay per action
USE_PYNPUT = True

if USE_PYNPUT:
    from pynput.mouse import Button, Controller as MouseController
    _mouse = MouseController()

    def move_mouse(x, y):       _mouse.position = (x, y)
    def click_mouse(b="left"):  _mouse.click(Button.left if b == "left" else Button.right)
    def scroll_mouse(dy):       _mouse.scroll(0, dy)
else:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0

    def move_mouse(x, y):       pyautogui.moveTo(x, y)
    def click_mouse(b="left"):  pyautogui.click(button=b)
    def scroll_mouse(dy):       pyautogui.scroll(int(dy * 3))

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

DETECTION_CONF = 0.7
TRACKING_CONF  = 0.7
PRESENCE_CONF  = 0.7

# Smoothing: 0.0 = instant/raw, closer to 1.0 = very smooth but laggy
SMOOTHING = 0.2

# Pinch click threshold (fraction of frame width)
PINCH_THRESHOLD = 0.045

# Frames a gesture must be held before it fires again
CLICK_COOLDOWN_FRAMES = 8

# Active region of the frame mapped to the full screen
ACTIVE_AREA = (0.10, 0.10, 0.90, 0.90)

# ── Landmark indices ──────────────────────────────────────────────────────────
# Same numbering in both old and new MediaPipe API
WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
INDEX_PIP  = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP   = 16
RING_PIP   = 14

# ── Screen dimensions ─────────────────────────────────────────────────────────
import tkinter as tk
_root = tk.Tk()
SCREEN_W = _root.winfo_screenwidth()
SCREEN_H = _root.winfo_screenheight()
_root.destroy()

# ── Model download ────────────────────────────────────────────────────────────

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[INFO] Downloading model to '{MODEL_PATH}' ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Download complete.")

# ── Helper functions ──────────────────────────────────────────────────────────

def lm_px(landmarks, idx, w, h):
    """Convert a normalised NormalizedLandmark to pixel coords."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def finger_up(landmarks, tip_idx, pip_idx):
    """True when finger tip is above (smaller y) its PIP joint → finger extended."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def map_to_screen(x_px, y_px, frame_w, frame_h):
    al, at, ar, ab = ACTIVE_AREA
    x_norm = (x_px / frame_w - al) / (ar - al)
    y_norm = (y_px / frame_h - at) / (ab - at)
    sx = int(np.clip(x_norm * SCREEN_W, 0, SCREEN_W - 1))
    sy = int(np.clip(y_norm * SCREEN_H, 0, SCREEN_H - 1))
    return sx, sy


def smooth(prev, curr, alpha=SMOOTHING):
    """Exponential moving average for cursor coordinates."""
    return (
        int(prev[0] * (1 - alpha) + curr[0] * alpha),
        int(prev[1] * (1 - alpha) + curr[1] * alpha),
    )

# ── Manual skeleton drawing ───────────────────────────────────────────────────
# The Tasks API has no built-in draw_landmarks helper like the legacy API did,
# so we draw connections manually.

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),          # palm cross-connections
]

def draw_skeleton(frame, landmarks, w, h, color=(0, 220, 110)):
    pts = [lm_px(landmarks, i, w, h) for i in range(21)]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1)

# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_hud(frame, gesture, cursor_pos, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Gesture : {gesture}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)
    cv2.putText(frame, f"Cursor  : {cursor_pos[0]}, {cursor_pos[1]}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)
    cv2.putText(frame, f"FPS     : {fps:.1f}",
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)


def draw_active_area(frame, w, h):
    al, at, ar, ab = ACTIVE_AREA
    cv2.rectangle(frame,
                  (int(al * w), int(at * h)),
                  (int(ar * w), int(ab * h)),
                  (80, 80, 80), 1)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ensure_model()

    # ── HandLandmarker setup (current Tasks API) ──────────────────────────────
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,      # VIDEO mode for webcam streams
        num_hands=1,
        min_hand_detection_confidence=DETECTION_CONF,
        min_hand_presence_confidence=PRESENCE_CONF,
        min_tracking_confidence=TRACKING_CONF,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        return

    prev_screen_pos = (SCREEN_W // 2, SCREEN_H // 2)
    click_cooldown  = 0
    scroll_prev_y   = None
    prev_time       = time.time()

    print("[INFO] Hand mouse controller running.")
    print("       Index finger up          → move cursor")
    print("       Pinch (index + thumb)    → left click")
    print("       Index + Middle           → scroll")
    print("       Index + Middle + Ring    → right click")
    print("       Press q or ESC to quit.\n")

    # Context manager ensures the landmarker is properly shut down on exit
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame not received — retrying...")
                continue

            frame = cv2.flip(frame, 1)   # mirror so movement feels natural
            h, w  = frame.shape[:2]

            # Wrap the frame in mp.Image (required by the Tasks API)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode requires a monotonically-increasing timestamp in ms
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture    = "none"
            cursor_pos = prev_screen_pos

            # result.hand_landmarks → list of hands
            # Each hand is a list of 21 NormalizedLandmark objects
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]   # first hand

                draw_skeleton(frame, landmarks, w, h)

                index_up  = finger_up(landmarks, INDEX_TIP,  INDEX_PIP)
                middle_up = finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
                ring_up   = finger_up(landmarks, RING_TIP,   RING_PIP)

                index_px = lm_px(landmarks, INDEX_TIP, w, h)
                thumb_px = lm_px(landmarks, THUMB_TIP, w, h)
                pinch    = distance(index_px, thumb_px) / w   # normalised

                # ── Gesture dispatch ──────────────────────────────────────────

                if index_up and middle_up and ring_up:
                    gesture = "right_click"
                    if click_cooldown == 0:
                        click_mouse("right")
                        click_cooldown = CLICK_COOLDOWN_FRAMES
                    scroll_prev_y = None

                elif index_up and middle_up:
                    gesture = "scroll"
                    mid_px  = lm_px(landmarks, MIDDLE_TIP, w, h)
                    avg_y   = (index_px[1] + mid_px[1]) / 2
                    if scroll_prev_y is not None:
                        dy = (scroll_prev_y - avg_y) / h
                        if abs(dy) > 0.003:          # dead-zone
                            scroll_mouse(dy * 10)
                    scroll_prev_y = avg_y

                elif index_up:
                    scroll_prev_y = None
                    raw        = map_to_screen(index_px[0], index_px[1], w, h)
                    cursor_pos = smooth(prev_screen_pos, raw)
                    move_mouse(*cursor_pos)
                    prev_screen_pos = cursor_pos

                    if pinch < PINCH_THRESHOLD:
                        gesture = "click"
                        if click_cooldown == 0:
                            click_mouse("left")
                            click_cooldown = CLICK_COOLDOWN_FRAMES
                    else:
                        gesture = "move"

                # Pinch indicator line between index and thumb
                color = (0, 80, 255) if pinch < PINCH_THRESHOLD else (0, 230, 120)
                cv2.line(frame, index_px, thumb_px, color, 2)
                cv2.circle(frame, index_px, 7, color, -1)
                cv2.circle(frame, thumb_px, 7, color, -1)

            if click_cooldown > 0:
                click_cooldown -= 1

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_active_area(frame, w, h)
            draw_hud(frame, gesture, cursor_pos, fps)

            cv2.imshow("Hand Mouse Control  |  q / ESC to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    if "--download" in sys.argv:
        ensure_model()
    else:
        main()