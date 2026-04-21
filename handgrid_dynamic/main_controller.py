from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "dynamic_gestures"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from ocsort import (  # type: ignore
    KalmanBoxTracker,
    associate,
    ciou_batch,
    ct_dist,
    diou_batch,
    giou_batch,
    iou_batch,
    linear_assignment,
)

from .onnx_models import HandClassification, HandDetection
from .utils import Deque, Hand

ASSO_FUNCS = {"iou": iou_batch, "giou": giou_batch, "ciou": ciou_batch, "diou": diou_batch, "ct_dist": ct_dist}


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


class MainController:
    def __init__(
        self,
        detection_model,
        classification_model,
        classification_metadata,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        maxlen=30,
        min_frames=5,
        detection_threshold=0.5,
    ):
        self.maxlen = maxlen
        self.min_frames = min_frames
        self.max_age = max_age
        self.min_hits = min_hits
        self.delta_t = 3
        self.iou_threshold = iou_threshold
        self.inertia = 0.2
        self.asso_func = ASSO_FUNCS["giou"]
        self.tracks = []
        self.frame_count = 0
        self.detection_threshold = detection_threshold
        self.detection_model = HandDetection(detection_model)
        self.classification_model = HandClassification(classification_model, classification_metadata)

    def update(self, dets=np.empty((0, 5)), labels=None, scores=None):
        if len(dets) == 0:
            for trk in self.tracks:
                trk["hands"].mark_missing()
            return

        self.frame_count += 1
        trks = np.zeros((len(self.tracks), 5))
        to_del = []
        ret = []
        lbs = []
        confs = []
        for t, trk in enumerate(trks):
            pos = self.tracks[t]["tracker"].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)

        velocities = np.array(
            [trk["tracker"].velocity if trk["tracker"].velocity is not None else np.array((0, 0)) for trk in self.tracks]
        )
        last_boxes = np.array([trk["tracker"].last_observation for trk in self.tracks])
        k_observations = np.array(
            [k_previous_obs(trk["tracker"].observations, trk["tracker"].age, self.delta_t) for trk in self.tracks]
        )

        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
        )

        for m in matched:
            self.tracks[m[1]]["tracker"].update(dets[m[0], :])
            self.tracks[m[1]]["hands"].append(Hand(bbox=dets[m[0], :4], gesture_name=labels[m[0]], score=scores[m[0]]))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = np.array(self.asso_func(left_dets, left_trks))
            if iou_left.size and iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.tracks[trk_ind]["tracker"].update(dets[det_ind, :])
                    self.tracks[trk_ind]["hands"].append(
                        Hand(bbox=dets[det_ind, :4], gesture_name=labels[det_ind], score=scores[det_ind])
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.tracks[m]["tracker"].update(None)
            self.tracks[m]["hands"].mark_missing()

        for i in unmatched_dets:
            hand_history = Deque(self.maxlen, self.min_frames)
            hand_history.append(Hand(bbox=dets[i, :4], gesture_name=labels[i], score=scores[i]))
            self.tracks.append(
                {
                    "hands": hand_history,
                    "tracker": KalmanBoxTracker(dets[i, :], delta_t=self.delta_t),
                }
            )

        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk["tracker"].last_observation[:4] if trk["tracker"].last_observation.sum() >= 0 else trk["tracker"].get_state()[0]
            if (trk["tracker"].time_since_update < 1) and (
                trk["tracker"].hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk["tracker"].id + 1])).reshape(1, -1))
                if len(trk["hands"]) > 0:
                    lbs.append(trk["hands"][-1].gesture_name)
                    confs.append(trk["hands"][-1].score)
                else:
                    lbs.append(None)
                    confs.append(None)

            i -= 1
            if trk["tracker"].time_since_update > self.max_age:
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret), lbs, confs
        return np.empty((0, 5)), [], []

    def __call__(self, frame):
        bboxes, probs = self.detection_model(frame)
        if len(bboxes) == 0:
            self.update(np.empty((0, 5)), None, None)
            return None, None, None, None

        probs = np.asarray(probs).reshape(-1)
        mask = probs >= self.detection_threshold
        bboxes = bboxes[mask]
        probs = probs[mask]
        if len(bboxes) == 0:
            self.update(np.empty((0, 5)), None, None)
            return None, None, None, None

        labels, class_scores, crop_boxes = self.classification_model(frame, bboxes)
        if not labels:
            self.update(np.empty((0, 5)), None, None)
            return None, None, None, None

        bboxes = np.asarray(crop_boxes, dtype=np.int32)
        probs = probs[: len(labels)]
        bboxes_with_scores = np.concatenate((bboxes, np.expand_dims(probs, axis=1)), axis=1)
        tracked_boxes, tracked_labels, tracked_scores = self.update(
            dets=bboxes_with_scores,
            labels=labels,
            scores=class_scores,
        )
        if tracked_boxes.shape[0] == 0:
            return None, None, None, None
        return tracked_boxes[:, :-1], tracked_boxes[:, -1], tracked_labels, tracked_scores
