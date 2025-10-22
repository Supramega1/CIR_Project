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
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os
import torch


def load_model(model_path: str = "yolov8s.pt", device: str = None) -> YOLO:
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
    """Draw bounding boxes and readable labels onto a copy of the frame and
    render the counts as a legend to the right of the image (outside the image).
    Returns the new image (wider) containing the original annotated image on the left
    and the legend on the right.
    """
    if result is None or not hasattr(result, "boxes"):
        # No results: still return an extended canvas with a small legend area
        h, w = frame.shape[:2]
        legend_w = 220
        new = np.full((h, w + legend_w, 3), 30, dtype=frame.dtype)  # dark background
        new[:, :w] = frame
        # draw "No objects detected" in legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        text = "No objects detected"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = w + 12
        y = 20 + th
        cv2.putText(new, text, (x, y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        return new

    boxes = result.boxes
    if boxes is None:
        return frame

    annotated = frame.copy()

    # fixed BGR color for boxes and a slightly darker bg color for label backgrounds
    fixed_color = (20, 120, 220)
    label_bg_color = tuple(max(0, int(c * 0.75)) for c in fixed_color)

    counts = {}
    total = 0

    for box in boxes:
        # robust bbox extraction
        try:
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
        except Exception:
            continue
        if xyxy.size < 4:
            continue
        x1, y1, x2, y2 = xyxy[:4]

        try:
            conf = float(box.conf.cpu().numpy())
        except Exception:
            conf = 0.0
        try:
            cls = int(box.cls.cpu().numpy())
        except Exception:
            cls = -1

        label = names.get(cls, str(cls)) if names is not None else str(cls)
        total += 1
        counts[label] = counts.get(label, 0) + 1

        text = f"{label}: {conf:.2f}"
        color = fixed_color

        # draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # draw label background and text with outline for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pad_x, pad_y = 6, 4
        lx1, ly1 = x1, max(0, y1 - th - pad_y * 2)
        lx2, ly2 = x1 + tw + pad_x * 2, y1
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), label_bg_color, -1)

        # text color chosen for contrast
        R, G, B = label_bg_color[2], label_bg_color[1], label_bg_color[0]
        luminance = 0.299 * R + 0.587 * G + 0.114 * B
        text_color = (0, 0, 0) if luminance > 150 else (255, 255, 255)
        outline_color = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)
        cv2.putText(annotated, text, (x1 + pad_x, y1 - pad_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (x1 + pad_x, y1 - pad_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Build legend lines (to go to the right of the image)
    if total == 0:
        lines = ["No objects detected"]
    else:
        lines = [f"Total: {total}", ""] + [f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]

    # compute legend size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_gap = 8
    widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    heights = [cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in lines]
    max_w = max(widths) if widths else 0
    total_h = sum(heights) + line_gap * (len(lines) - 1)

    right_pad = 12
    left_pad = 12
    top_pad = 12
    legend_w = max_w + left_pad + right_pad
    legend_h = max(total_h + 2 * top_pad, annotated.shape[0])

    # create new canvas wider than original to host legend at right
    h = max(annotated.shape[0], legend_h)
    new_w = annotated.shape[1] + legend_w
    new = np.full((h, new_w, 3), 30, dtype=annotated.dtype)  # dark background

    # place annotated image at left, centered vertically if legend taller
    y_offset = 0
    new[y_offset:y_offset + annotated.shape[0], :annotated.shape[1]] = annotated

    # draw legend background (a slightly lighter rectangle)
    lx1 = annotated.shape[1]
    ly1 = 0
    lx2 = new_w
    ly2 = h
    cv2.rectangle(new, (lx1, ly1), (lx2, ly2), (40, 40, 40), -1)
    cv2.rectangle(new, (lx1, ly1), (lx2 - 1, ly2 - 1), (80, 80, 80), 1)

    # draw each line in the legend
    cur_y = top_pad
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = annotated.shape[1] + left_pad
        text_y = cur_y + th
        if i == 0 and total != 0:
            # header style
            cv2.putText(new, line, (text_x, text_y), font, font_scale, (200, 200, 200), 2, cv2.LINE_AA)
        else:
            cv2.putText(new, line, (text_x, text_y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        cur_y = text_y + line_gap

    return new

def run_webcam(model_path: str = "yolo11n.pt", camera_index: int = 0, conf: float = 0.25):
    """Open the webcam and run real-time detection loop.

    Press 'q' to quit, 's' to save a screenshot (saved next to script).
    The loop will also exit if the display window is closed by the user.
    """
    model = load_model(model_path)
    names = getattr(model, "names", {}) or {}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    window_name = "Food detection - press q to quit, s to save"
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, stopping")
                break

            # resize for consistent speed/display
            frame = cv2.resize(frame, (640, 480))

            with torch.no_grad():
                result = predict_on_frame(model, frame, conf=conf)
            frame = draw_predictions(frame, result, names)

            # FPS calculation
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            # handle window closed by user (clicking the X)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Exiting.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
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

import matplotlib.pyplot as plt
def upload_and_show_image(model_path: str = "yolo11n.pt", conf: float = 0.25):
    """
    Repeatedly open a file dialog to choose an image, run the YOLO model on it,
    and display the resulting annotated image in a matplotlib window. After you
    close the matplotlib window, the file dialog appears again. The loop only
    ends when the file dialog is closed/cancelled (no file selected).
    """
    # load model once
    model = load_model(model_path)
    names = getattr(model, "names", {}) or {}

    # create a hidden root for filedialog and reuse it for each selection
    root = tk.Tk()
    root.withdraw()
    # keep dialog on top on some platforms
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        while True:
            file_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
                parent=root,
            )
            # user cancelled -> exit loop (terminate function)
            if not file_path:
                print("No file selected. Exiting image loop.")
                break

            # Normalize file URI if needed (tk may return file:///... on some platforms)
            if file_path.startswith("file:///"):
                # On Windows there may be an extra leading slash after file:///
                file_path = file_path[8:] if os.name == "nt" else file_path[7:]
            print(f"Selected file: {file_path}")

            # read image and run prediction
            img = cv2.imread(file_path)
            if img is None:
                # Fallback: use PIL which can handle more encodings/formats and unicode paths
                try:
                    pil_img = Image.open(file_path).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"Failed to load image: {file_path} ({e})")
                    # continue to next selection instead of raising
                    continue

            result = predict_on_frame(model, img, conf=conf)

            # draw predictions and show in a matplotlib figure
            annotated = draw_predictions(img.copy(), result, names)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 8))
            plt.imshow(annotated_rgb)
            plt.axis("off")
            plt.title(f"Predictions - {os.path.basename(file_path)}")
            plt.show()  # blocks until the figure window is closed; then loop repeats

    finally:
        try:
            root.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    args = parse_args()
    run_webcam(args.model, args.camera, args.conf)
    #upload_and_show_image(args.model, args.conf)
