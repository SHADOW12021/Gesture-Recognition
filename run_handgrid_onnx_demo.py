from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller as MouseController

from handgrid_dynamic import MainController
from handgrid_dynamic.utils import Event

MOVE_GESTURES = {"one"}
SCROLL_GESTURES = {"two_up", "two_up_inverted"}
ZOOM_IN_GESTURES = {"thumb_index"}
ZOOM_OUT_GESTURES = {"thumb_index2"}
CLICK_GESTURES = {"ok"}
RIGHT_CLICK_GESTURES = {"three"}
PAUSE_TOGGLE_GESTURES = {"call"}
DRAG_GESTURES = {"fist"}

ACTIVE_AREA = (0.05, 0.05, 0.95, 0.95)
CURSOR_SMOOTHING = 0.28
SCROLL_GAIN = 45.0
SCROLL_DEADZONE = 0.012
GESTURE_COOLDOWN_SECONDS = 0.45

mouse = MouseController()
keyboard = KeyboardController()
_root = tk.Tk()
SCREEN_W = _root.winfo_screenwidth()
SCREEN_H = _root.winfo_screenheight()
_root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX-based dynamic gesture demo close to dynamic_gestures.")
    parser.add_argument(
        "--detector",
        default="dynamic_gestures/models/hand_detector.onnx",
        help="Path to the ONNX hand detector. The subset does not include detection labels, so this is reused or swapped in.",
    )
    parser.add_argument("--classifier", required=True, help="Path to the exported ONNX gesture classifier.")
    parser.add_argument("--metadata", default=None, help="Path to the classifier metadata JSON file. Defaults to classifier basename with .json.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    return parser.parse_args()


def event_text(event: Event) -> str:
    return event.name.replace("_", " ").title()


def map_to_screen(x_px: float, y_px: float, frame_w: int, frame_h: int) -> tuple[int, int]:
    left, top, right, bottom = ACTIVE_AREA
    x_norm = (x_px / frame_w - left) / (right - left)
    y_norm = (y_px / frame_h - top) / (bottom - top)
    sx = int(np.clip(x_norm * SCREEN_W, 0, SCREEN_W - 1))
    sy = int(np.clip(y_norm * SCREEN_H, 0, SCREEN_H - 1))
    return sx, sy


def smooth_position(previous: tuple[int, int], current: tuple[int, int], alpha: float = CURSOR_SMOOTHING) -> tuple[int, int]:
    return (
        int(previous[0] * (1 - alpha) + current[0] * alpha),
        int(previous[1] * (1 - alpha) + current[1] * alpha),
    )


def draw_active_area(frame) -> None:
    height, width = frame.shape[:2]
    left, top, right, bottom = ACTIVE_AREA
    cv2.rectangle(
        frame,
        (int(left * width), int(top * height)),
        (int(right * width), int(bottom * height)),
        (80, 80, 80),
        1,
    )


class MouseGestureMapper:
    def __init__(self):
        self.prev_screen_pos = (SCREEN_W // 2, SCREEN_H // 2)
        self.prev_scroll_y = None
        self.zoom_cooldowns = {"in": 0.0, "out": 0.0}
        self.click_cooldown_until = 0.0
        self.right_click_cooldown_until = 0.0
        self.pause_toggle_cooldown_until = 0.0
        self.paused = False
        self.drag_active = False

    def update(self, frame, active_tracks: list[dict], now: float) -> tuple[str, tuple[int, int]]:
        gesture = "idle"
        cursor_pos = self.prev_screen_pos

        if not active_tracks:
            self.prev_scroll_y = None
            if self.drag_active:
                mouse.release(Button.left)
                self.drag_active = False
            return gesture, cursor_pos

        primary_track = max(active_tracks, key=lambda trk: trk["hands"][-1].score if len(trk["hands"]) > 0 else 0.0)
        hands = primary_track["hands"]
        if len(hands) == 0:
            self.prev_scroll_y = None
            return gesture, cursor_pos

        stable_label, votes = hands.stable_gesture()
        hand = hands[-1]
        if hand.center is None:
            self.prev_scroll_y = None
            if self.drag_active:
                mouse.release(Button.left)
                self.drag_active = False
            return gesture, cursor_pos

        if stable_label in PAUSE_TOGGLE_GESTURES and votes >= 3:
            self.prev_scroll_y = None
            if self.drag_active:
                mouse.release(Button.left)
                self.drag_active = False
            if now >= self.pause_toggle_cooldown_until:
                self.paused = not self.paused
                self.pause_toggle_cooldown_until = now + 0.8
            return ("paused" if self.paused else "resumed"), cursor_pos

        if self.paused:
            self.prev_scroll_y = None
            if self.drag_active:
                mouse.release(Button.left)
                self.drag_active = False
            return "paused", cursor_pos

        if stable_label in MOVE_GESTURES and votes >= 3:
            raw = map_to_screen(hand.center[0], hand.center[1], frame.shape[1], frame.shape[0])
            cursor_pos = smooth_position(self.prev_screen_pos, raw)
            mouse.position = cursor_pos
            self.prev_screen_pos = cursor_pos
            self.prev_scroll_y = None
            gesture = "move_mouse"
        elif stable_label in SCROLL_GESTURES and votes >= 3:
            current_y = hand.center[1] / frame.shape[0]
            if self.prev_scroll_y is not None:
                delta = self.prev_scroll_y - current_y
                if abs(delta) > SCROLL_DEADZONE:
                    mouse.scroll(0, int(delta * SCROLL_GAIN))
                    gesture = "scroll_up" if delta > 0 else "scroll_down"
                else:
                    gesture = "scroll_hold"
            else:
                gesture = "scroll_hold"
            self.prev_scroll_y = current_y
        elif stable_label in ZOOM_IN_GESTURES and votes >= 3:
            self.prev_scroll_y = None
            if now >= self.zoom_cooldowns["in"]:
                with keyboard.pressed(Key.ctrl):
                    mouse.scroll(0, 2)
                self.zoom_cooldowns["in"] = now + GESTURE_COOLDOWN_SECONDS
            gesture = "zoom_in"
        elif stable_label in ZOOM_OUT_GESTURES and votes >= 3:
            self.prev_scroll_y = None
            if now >= self.zoom_cooldowns["out"]:
                with keyboard.pressed(Key.ctrl):
                    mouse.scroll(0, -2)
                self.zoom_cooldowns["out"] = now + GESTURE_COOLDOWN_SECONDS
            gesture = "zoom_out"
        elif stable_label in CLICK_GESTURES and votes >= 3:
            self.prev_scroll_y = None
            if now >= self.click_cooldown_until:
                mouse.click(Button.left, 1)
                self.click_cooldown_until = now + GESTURE_COOLDOWN_SECONDS
            gesture = "left_click"
        elif stable_label in RIGHT_CLICK_GESTURES and votes >= 3:
            self.prev_scroll_y = None
            if now >= self.right_click_cooldown_until:
                mouse.click(Button.right, 1)
                self.right_click_cooldown_until = now + GESTURE_COOLDOWN_SECONDS
            gesture = "right_click"
        elif stable_label in DRAG_GESTURES and votes >= 3:
            raw = map_to_screen(hand.center[0], hand.center[1], frame.shape[1], frame.shape[0])
            cursor_pos = smooth_position(self.prev_screen_pos, raw)
            mouse.position = cursor_pos
            self.prev_screen_pos = cursor_pos
            self.prev_scroll_y = None
            if not self.drag_active:
                mouse.press(Button.left)
                self.drag_active = True
            gesture = "drag_hold"
        else:
            self.prev_scroll_y = None
            if self.drag_active:
                mouse.release(Button.left)
                self.drag_active = False
            gesture = stable_label if stable_label != "no_gesture" else "idle"

        return gesture, cursor_pos


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata or str(Path(args.classifier).with_suffix(".json"))
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    controller = MainController(
        detection_model=args.detector,
        classification_model=args.classifier,
        classification_metadata=metadata_path,
        detection_threshold=args.detection_threshold,
    )
    mouse_mapper = MouseGestureMapper()

    banner_text = ""
    banner_until = 0.0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        start_time = time.time()
        bboxes, ids, labels, scores = controller(frame)
        active_tracks = [trk for trk in controller.tracks if trk["tracker"].time_since_update < 1 and len(trk["hands"]) > 0]
        current_gesture, cursor_pos = mouse_mapper.update(frame, active_tracks, time.time())

        if bboxes is not None:
            bboxes = bboxes.astype(np.int32)
            for i in range(bboxes.shape[0]):
                box = bboxes[i, :]
                label = labels[i] if labels[i] is not None else "None"
                score = scores[i] if scores[i] is not None else 0.0
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 3)
                cv2.putText(
                    frame,
                    f"ID {int(ids[i])}: {label} {score:.2f}",
                    (box[0], max(25, box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                        (0, 0, 255),
                        2,
                )

        for trk in controller.tracks:
            if trk["tracker"].time_since_update < 1 and trk["hands"].action is not None:
                banner_text = event_text(trk["hands"].action)
                banner_until = time.time() + 1.0
                print(banner_text)
                trk["hands"].clear_action()

        if time.time() < banner_until:
            cv2.putText(frame, banner_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 255), 3)

        cv2.putText(frame, f"mouse action: {current_gesture}", (20, 45 if time.time() >= banner_until else 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 120), 2)

        if args.debug:
            fps = 1.0 / max(1e-6, time.time() - start_time)
            draw_active_area(frame)
            cv2.putText(frame, f"fps {fps:.2f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"cursor {cursor_pos[0]}, {cursor_pos[1]}", (20, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            for trk in controller.tracks:
                if trk["tracker"].time_since_update < 1 and len(trk["hands"]) > 0:
                    hand = trk["hands"][-1]
                    if hand.center is not None:
                        center = tuple(np.int32(hand.center))
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        cv2.imshow("HandGrid ONNX Dynamic Demo", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
