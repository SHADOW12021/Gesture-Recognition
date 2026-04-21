from __future__ import annotations


class Hand:
    def __init__(self, bbox, gesture_name=None, score=0.0):
        self.bbox = bbox
        self.gesture_name = gesture_name
        self.score = score
        self.position = None
        if bbox is not None:
            self.center = self._get_center()
            self.size = max(1.0, max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        else:
            self.center = None
            self.size = 1.0

    def _get_center(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def __repr__(self):
        return f"Hand(center={self.center}, size={self.size}, gesture={self.gesture_name}, score={self.score:.3f})"
