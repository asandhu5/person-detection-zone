import streamlit as st
from ultralytics import YOLO

PERSON_CLASS_ID = 0

MODEL_OPTIONS: dict[str, str] = {
    "YOLOv10n — Fastest  (6 MB)":  "yolov10n.pt",
    "YOLOv10s — Balanced (16 MB)": "yolov10s.pt",
    "YOLOv10m — Accurate (32 MB)": "yolov10m.pt",
}


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """
    Load a YOLO model and cache it for the entire Streamlit session.
    First call downloads weights; all subsequent calls return cached object.

    Args:
        model_path: e.g. "yolov10n.pt"
    Returns:
        Loaded YOLO model instance
    """
    return YOLO(model_path)


def parse_detections(results, confidence_threshold: float) -> list[dict]:
    """
    Convert raw ultralytics Results into clean Python dicts.
    Separated so it can be tested without a real model.

    Args:
        results:              ultralytics Results object (results[0])
        confidence_threshold: minimum confidence to keep
    Returns:
        [{"box": [x1,y1,x2,y2], "confidence": float}, ...]
    """
    detections: list[dict] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id == PERSON_CLASS_ID and conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "box":        [x1, y1, x2, y2],
                "confidence": round(conf, 2),
            })
    return detections


def detect_persons(model: YOLO, frame, confidence_threshold: float = 0.45) -> list[dict]:
    """
    Run YOLOv10 on a single video frame and return all person detections.

    Args:
        model:                loaded YOLO model
        frame:                BGR numpy array (OpenCV format)
        confidence_threshold: minimum confidence score
    Returns:
        [{"box": [x1,y1,x2,y2], "confidence": float}, ...]
    """
    results = model(frame, verbose=False)[0]
    return parse_detections(results, confidence_threshold)
