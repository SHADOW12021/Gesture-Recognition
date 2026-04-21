from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

from .constants import CLOSED_GESTURES, OPEN_GESTURES, SWIPE_GESTURES


@dataclass
class DynamicGestureEvent:
    name: str
    hand_id: str
    label: str
    confidence: float


@dataclass
class HandObservation:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    timestamp: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def hand_size(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(1.0, float(max(x2 - x1, y2 - y1)))


class DynamicGestureController:
    def __init__(
        self,
        hand_id: str,
        label_window: int = 6,
        stable_votes: int = 4,
        min_confidence: float = 0.45,
        drag_hold_frames: int = 5,
        release_hold_frames: int = 3,
        swipe_timeout_seconds: float = 1.0,
        cooldown_seconds: float = 0.5,
    ):
        self.hand_id = hand_id
        self.label_window = label_window
        self.stable_votes = stable_votes
        self.min_confidence = min_confidence
        self.drag_hold_frames = drag_hold_frames
        self.release_hold_frames = release_hold_frames
        self.swipe_timeout_seconds = swipe_timeout_seconds
        self.cooldown_seconds = cooldown_seconds

        self.history: deque[HandObservation] = deque(maxlen=32)
        self.label_history: deque[str] = deque(maxlen=label_window)
        self.cooldown_until = 0.0
        self.dragging = False
        self.drag_anchor: Optional[tuple[float, float]] = None
        self.swipe_candidate: Optional[dict] = None
        self.active_label = "no_gesture"

    def update(
        self,
        label: str,
        confidence: float,
        bbox: tuple[int, int, int, int],
        timestamp: float,
    ) -> list[DynamicGestureEvent]:
        observation = HandObservation(label=label, confidence=confidence, bbox=bbox, timestamp=timestamp)
        self.history.append(observation)
        self.label_history.append(label if confidence >= self.min_confidence else "no_gesture")

        stable_label, stable_count = self._stable_label()
        self.active_label = stable_label
        events: list[DynamicGestureEvent] = []

        if timestamp < self.cooldown_until:
            if self.dragging:
                events.extend(self._handle_drag(stable_label, stable_count, observation))
            return events

        events.extend(self._handle_drag(stable_label, stable_count, observation))
        events.extend(self._handle_swipe(stable_label, stable_count, observation))
        return events

    def mark_missing(self, timestamp: float) -> None:
        if self.dragging and self.history and timestamp - self.history[-1].timestamp > 0.75:
            self.dragging = False
            self.drag_anchor = None
        if self.swipe_candidate and timestamp - self.swipe_candidate["start_time"] > self.swipe_timeout_seconds:
            self.swipe_candidate = None
        self.label_history.append("no_gesture")
        self.active_label = "no_gesture"

    def _stable_label(self) -> tuple[str, int]:
        if not self.label_history:
            return "no_gesture", 0
        counts = Counter(self.label_history)
        label, count = counts.most_common(1)[0]
        if count < self.stable_votes:
            return "no_gesture", count
        return label, count

    def _handle_drag(
        self,
        stable_label: str,
        stable_count: int,
        observation: HandObservation,
    ) -> list[DynamicGestureEvent]:
        events: list[DynamicGestureEvent] = []

        if not self.dragging and stable_label in CLOSED_GESTURES and stable_count >= self.drag_hold_frames:
            self.dragging = True
            self.drag_anchor = observation.center
            events.append(self._event("drag_start", stable_label, observation.confidence))
            return events

        if self.dragging and stable_label in OPEN_GESTURES and stable_count >= self.release_hold_frames:
            self.dragging = False
            self.drag_anchor = None
            self.cooldown_until = observation.timestamp + self.cooldown_seconds
            events.append(self._event("drop", stable_label, observation.confidence))

        return events

    def _handle_swipe(
        self,
        stable_label: str,
        stable_count: int,
        observation: HandObservation,
    ) -> list[DynamicGestureEvent]:
        events: list[DynamicGestureEvent] = []

        if stable_label in SWIPE_GESTURES and stable_count >= self.release_hold_frames:
            if self.swipe_candidate is None:
                self.swipe_candidate = {
                    "label": stable_label,
                    "start_center": observation.center,
                    "start_size": observation.hand_size,
                    "start_time": observation.timestamp,
                    "max_distance": 0.0,
                    "direction": None,
                }
            else:
                dx = observation.center[0] - self.swipe_candidate["start_center"][0]
                dy = observation.center[1] - self.swipe_candidate["start_center"][1]
                distance = (dx * dx + dy * dy) ** 0.5
                self.swipe_candidate["max_distance"] = max(self.swipe_candidate["max_distance"], distance)
                self.swipe_candidate["direction"] = _resolve_direction(dx, dy)
            return events

        if self.swipe_candidate is None:
            return events

        elapsed = observation.timestamp - self.swipe_candidate["start_time"]
        distance_ratio = self.swipe_candidate["max_distance"] / max(1.0, self.swipe_candidate["start_size"])
        direction = self.swipe_candidate["direction"]

        if (
            stable_label in OPEN_GESTURES
            and stable_count >= self.release_hold_frames
            and elapsed <= self.swipe_timeout_seconds
            and distance_ratio >= 1.1
            and direction is not None
        ):
            self.cooldown_until = observation.timestamp + self.cooldown_seconds
            events.append(self._event(f"swipe_{direction}", stable_label, observation.confidence))

        if elapsed > self.swipe_timeout_seconds or stable_label in OPEN_GESTURES or stable_label in CLOSED_GESTURES:
            self.swipe_candidate = None

        return events

    def _event(self, name: str, label: str, confidence: float) -> DynamicGestureEvent:
        return DynamicGestureEvent(name=name, hand_id=self.hand_id, label=label, confidence=confidence)


def _resolve_direction(dx: float, dy: float) -> Optional[str]:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
