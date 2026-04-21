from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from gesture_pipeline.constants import IMAGENET_MEAN, IMAGENET_STD


class OnnxModel:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]


class HandDetection(OnnxModel):
    def __init__(self, model_path: str | Path, image_size: tuple[int, int] = (320, 240)):
        super().__init__(model_path)
        self.image_size = image_size
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image.astype(np.float32), axis=0)

    def __call__(self, frame: np.ndarray):
        input_tensor = self.preprocess(frame)
        boxes, _, probs = self.session.run(self.output_names, {self.input_name: input_tensor})
        height, width = frame.shape[:2]
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return boxes.astype(np.int32), probs.astype(np.float32)


class HandClassification(OnnxModel):
    def __init__(self, model_path: str | Path, metadata_path: str | Path):
        super().__init__(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.class_names = metadata["class_names"]
        self.image_size = int(metadata["image_size"])
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    @staticmethod
    def get_square(box, image_shape):
        height, width = image_shape[:2]
        x0, y0, x1, y1 = map(int, box)
        w, h = x1 - x0, y1 - y0
        if h < w:
            y0 -= int((w - h) / 2)
            y1 = y0 + w
        elif h > w:
            x0 -= int((h - w) / 2)
            x1 = x0 + h
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width - 1, x1)
        y1 = min(height - 1, y1)
        return x0, y0, x1, y1

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image.astype(np.float32), axis=0)

    def get_crops(self, frame: np.ndarray, bboxes: np.ndarray):
        crops = []
        crop_boxes = []
        for bbox in bboxes:
            x0, y0, x1, y1 = self.get_square(bbox, frame.shape)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crops.append(self.preprocess(crop))
            crop_boxes.append((x0, y0, x1, y1))
        if not crops:
            return None, []
        return np.concatenate(crops, axis=0), crop_boxes

    def __call__(self, frame: np.ndarray, bboxes: np.ndarray):
        inputs, crop_boxes = self.get_crops(frame, bboxes)
        if inputs is None:
            return [], [], []
        logits = self.session.run(self.output_names, {self.input_name: inputs})[0]
        probs = _softmax(logits)
        label_indices = np.argmax(probs, axis=1)
        scores = probs[np.arange(len(label_indices)), label_indices]
        labels = [self.class_names[int(index)] for index in label_indices]
        return labels, scores.tolist(), crop_boxes


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)
