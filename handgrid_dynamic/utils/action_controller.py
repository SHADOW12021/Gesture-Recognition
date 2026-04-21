from __future__ import annotations

from collections import Counter, deque
from math import hypot

from .enums import Event
from .hand import Hand

OPEN_GESTURES = {"palm", "stop", "stop_inverted", "no_gesture"}
SWIPE_GESTURES = {"point", "one", "two_up", "two_up_inverted", "peace", "peace_inverted"}


class Deque:
    def __init__(self, maxlen=30, min_frames=5):
        self.maxlen = maxlen
        self.min_frames = min_frames
        self._deque = []
        self.action = None
        self.label_deque = deque(maxlen=maxlen)
        self.swipe_candidate = None
        self.cooldown = 0

    def __len__(self):
        return len(self._deque)

    def __getitem__(self, index):
        return self._deque[index]

    def __iter__(self):
        return iter(self._deque)

    def append(self, hand: Hand):
        if self.maxlen is not None and len(self._deque) >= self.maxlen:
            self._deque.pop(0)
        self._deque.append(hand)
        self.label_deque.append(hand.gesture_name if hand.gesture_name is not None else "no_gesture")
        self.check_is_action(hand)

    def mark_missing(self):
        self.label_deque.append("no_gesture")
        if self.swipe_candidate is not None:
            self.swipe_candidate["misses"] += 1
            if self.swipe_candidate["misses"] > self.min_frames:
                self.swipe_candidate = None

    def stable_gesture(self, count=None):
        if not self.label_deque:
            return "no_gesture", 0
        if count is None:
            count = min(self.min_frames, len(self.label_deque))
        recent = list(self.label_deque)[-count:]
        label, votes = Counter(recent).most_common(1)[0]
        return label, votes

    def check_is_action(self, hand: Hand):
        label, votes = self.stable_gesture()
        if self.cooldown > 0:
            self.cooldown -= 1
            return False

        if label in SWIPE_GESTURES and hand.center is not None:
            if self.swipe_candidate is None:
                self.swipe_candidate = {
                    "label": label,
                    "start_center": hand.center,
                    "start_size": hand.size,
                    "misses": 0,
                }
            else:
                self.swipe_candidate["misses"] = 0
            return False

        if self.swipe_candidate and label in OPEN_GESTURES and hand.center is not None:
            dx = hand.center[0] - self.swipe_candidate["start_center"][0]
            dy = hand.center[1] - self.swipe_candidate["start_center"][1]
            distance_ratio = hypot(dx, dy) / max(1.0, self.swipe_candidate["start_size"])
            if distance_ratio >= 1.1:
                if abs(dx) >= abs(dy):
                    self.action = Event.SWIPE_RIGHT if dx > 0 else Event.SWIPE_LEFT
                else:
                    self.action = Event.SWIPE_DOWN if dy > 0 else Event.SWIPE_UP
                self.swipe_candidate = None
                self.cooldown = self.min_frames
                return True
            self.swipe_candidate = None

        return False

    def clear_action(self):
        self.action = None
