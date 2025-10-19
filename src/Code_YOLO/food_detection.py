"""Real-time food detection using an Ultralytics YOLO model

This script provides small, well-factored functions:
- load_model(model_path, device)
- predict_on_frame(model, frame, conf)
- draw_predictions(frame, result, names)
- run_webcam(model_path, camera_index, conf)

By default it will try to load `yolo11n.pt` from the repository root. If you exported a Roboflow YOLOv8 model, point
`--model` to the .pt file you downloaded.
"""

import argparse
import time
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path: str = "yolo11n.pt", device: str = None) -> YOLO:
    """Load a YOLO model from a weights file or a model string.

    Args:
        model_path: path to the weights file (e.g. 'yolo11n.pt') or model spec supported by ultralytics.
        device: optional device string like 'cpu' or '0' for GPU. If None the library decides.

    Returns:
        An instantiated ultralytics.YOLO object.
    """
    if device:
        model = YOLO(model_path, device=device)
    else:
        model = YOLO(model_path)
    try:
        model.info()  # prints model architecture and class names
    except Exception:
        # info is optional; ignore failures to keep runtime robust
        pass
    return model


def predict_on_frame(model: YOLO, frame: np.ndarray, conf: float = 0.25):
    """Run model inference on a single BGR frame.

    Returns the ultralytics Results object for the frame.
    """
    # Ultralyics accepts numpy arrays (BGR) directly.
    # We request a single-frame prediction with the provided confidence threshold.
    results = model(frame, conf=conf, verbose=False)
    # results is an iterable; return the first result for convenience
    return results[0]


def draw_predictions(frame: np.ndarray, result, names: dict) -> np.ndarray:
    """Draw bounding boxes and labels onto the frame in-place and return it.

    Args:
        frame: OpenCV BGR image.
        result: a single ultralytics Result (for one image).
        names: mapping from class id to class name (model.names).
    """
    if result is None or not hasattr(result, "boxes"):
        return frame

    boxes = result.boxes
    if boxes is None:
        return frame

    for box in boxes:
        # box.xyxy -> tensor or ndarray with shape (4,)
        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        x1, y1, x2, y2 = xyxy[:4]
        conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else 0.0
        cls = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else -1
        label = names.get(cls, str(cls)) if names is not None else str(cls)
        text = f"{label}: {conf:.2f}"

        # box color: pick deterministic color from class id
        color = tuple(int(c) for c in (np.array([37, 99, 140]) + (cls * 23) % 200) ) if cls >= 0 else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # label background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def run_webcam(model_path: str = "yolo11n.pt", camera_index: int = 0, conf: float = 0.25):
    """Open the webcam and run real-time detection loop.

    Press 'q' to quit, 's' to save a screenshot (saved next to script).
    """
    model = load_model(model_path)

    names = getattr(model, "names", {}) or {}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, stopping")
                break

            frame_count += 1
            # Run prediction. For speed you could skip frames or lower imgsz/conf.
            result = predict_on_frame(model, frame, conf=conf)
            draw_predictions(frame, result, names)

            # FPS calculation
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Food detection - press q to quit, s to save", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                # save screenshot
                filename = f"food_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run real-time food detection using a YOLO model and your webcam")
    p.add_argument("--model", "-m", default="yolo11n.pt", help="Path to YOLO weights (.pt) or model spec")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index for cv2.VideoCapture")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_webcam(args.model, args.camera, args.conf)
