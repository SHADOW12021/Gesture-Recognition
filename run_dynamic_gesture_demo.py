from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from PIL import Image
from torchvision import transforms

from gesture_pipeline.checkpoints import load_checkpoint
from gesture_pipeline.constants import IMAGENET_MEAN, IMAGENET_STD
from gesture_pipeline.dynamic_logic import DynamicGestureController
from gesture_pipeline.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time static + dynamic hand gesture recognition.")
    parser.add_argument("--checkpoint", default="checkpoints/static_gesture_model.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_model(checkpoint_path: str | Path, device: str):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = create_model(checkpoint["architecture"], len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["class_names"], checkpoint["image_size"]


def expand_bbox(bbox, frame_shape, scale: float = 0.2):
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * scale)
    pad_y = int(h * scale)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    )


def landmarks_to_bbox(hand_landmarks, frame_shape):
    height, width = frame_shape[:2]
    xs = [landmark.x * width for landmark in hand_landmarks.landmark]
    ys = [landmark.y * height for landmark in hand_landmarks.landmark]
    return (
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys)),
    )


@torch.no_grad()
def classify_crop(frame, bbox, model, transform, class_names, device):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "no_gesture", 0.0
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    score, label_idx = probs.max(dim=0)
    return class_names[label_idx.item()], score.item()


def main() -> None:
    args = parse_args()
    model, class_names, image_size = load_model(args.checkpoint, args.device)
    transform = build_transform(image_size)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    controllers = {
        "Left": DynamicGestureController("Left"),
        "Right": DynamicGestureController("Right"),
    }

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_banner = ""
    banner_expires_at = 0.0
    previous_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        seen_hands = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_id = handedness.classification[0].label
                seen_hands.add(hand_id)
                bbox = expand_bbox(landmarks_to_bbox(hand_landmarks, frame.shape), frame.shape)
                label, confidence = classify_crop(frame, bbox, model, transform, class_names, args.device)
                events = controllers[hand_id].update(label, confidence, bbox, now)

                x1, y1, x2, y2 = bbox
                color = (0, 220, 0) if controllers[hand_id].dragging else (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{hand_id}: {label} {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if controllers[hand_id].dragging:
                    cv2.putText(
                        frame,
                        "state: dragging",
                        (x1, min(frame.shape[0] - 20, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                    )

                for event in events:
                    current_banner = f"{event.hand_id} -> {event.name.upper()}"
                    banner_expires_at = now + 1.0
                    print(current_banner)

        for hand_id, controller in controllers.items():
            if hand_id not in seen_hands:
                controller.mark_missing(now)

        if now < banner_expires_at:
            cv2.putText(
                frame,
                current_banner,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (30, 30, 255),
                3,
            )

        fps = 1.0 / max(1e-6, now - previous_time)
        previous_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dynamic Gesture Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
