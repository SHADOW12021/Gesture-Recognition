from pathlib import Path
import shutil
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
BACKUP_MODEL_PATH = Path(__file__).resolve().parents[1] / "ios" / "GestureRecognizer" / "gesture_recognizer.task"

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (88, 205, 54)


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    if not BACKUP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find a valid gesture model at {MODEL_PATH} or {BACKUP_MODEL_PATH}."
        )
    shutil.copyfile(BACKUP_MODEL_PATH, MODEL_PATH)
    return MODEL_PATH


def draw_recognition_result(image_bgr, recognition_result):
    annotated_image = image_bgr.copy()
    for idx, hand_landmarks in enumerate(recognition_result.hand_landmarks):
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style(),
        )

        if recognition_result.gestures and recognition_result.gestures[idx]:
            top_gesture = recognition_result.gestures[idx][0]
            wrist = hand_landmarks[0]
            text_x = int(wrist.x * annotated_image.shape[1])
            text_y = max(MARGIN, int(wrist.y * annotated_image.shape[0]) - MARGIN)
            label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
            cv2.putText(
                annotated_image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
    return annotated_image


def main():
    model_path = ensure_model()
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. Try changing camera_index to 1."
        )

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        start_time = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            annotated_frame = draw_recognition_result(frame, recognition_result)

            cv2.imshow("MediaPipe Gesture Recognizer", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
