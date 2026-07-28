
import cv2
import numpy as np

Box     = list[int]               # [x1, y1, x2, y2]
Point   = tuple[int, int]         # (x, y) pixel coords
Polygon = list[tuple[int, int]]   # [(x,y), (x,y), ...]


def get_foot_point(box: Box) -> Point:
    """
    Return the bottom-center of a bounding box (where the person stands).
    More accurate than checking full-box overlap for zone detection.

    Args:
        box: [x1, y1, x2, y2]
    Returns:
        (cx, y2) — bottom-center pixel coordinate
    """
    x1, _, x2, y2 = box
    cx = int((x1 + x2) / 2)
    return (cx, int(y2))


def is_point_in_zone(point: Point, polygon: Polygon) -> bool:
    """
    Check whether a pixel point is inside (or on the edge of) a polygon zone.
    Uses OpenCV's pointPolygonTest (ray-casting algorithm).

    Args:
        point:   (x, y) to test
        polygon: list of (x, y) vertices
    Returns:
        True if inside or on edge, False if outside
    """
    pts    = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(
        pts, (float(point[0]), float(point[1])), measureDist=False
    )
    return result >= 0


def draw_zone(frame: np.ndarray, polygon: Polygon,
              border_color: tuple = (255, 50, 50),
              fill_alpha: float = 0.15) -> None:
    """
    Draw the restricted zone polygon on a frame in-place.
    Semi-transparent fill + solid border.

    Args:
        frame:        BGR numpy array (modified in-place)
        polygon:      list of (x, y) vertices
        border_color: BGR tuple for border
        fill_alpha:   fill transparency (0=invisible, 1=solid)
    """
    pts     = np.array(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (180, 30, 30))
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2)


def draw_detection(frame: np.ndarray, box: Box,
                   inside: bool, confidence: float,
                   show_conf: bool = True,
                   show_foot: bool = False) -> None:
    """
    Draw a single person bounding box on a frame in-place.
    Red if inside zone, green if outside.

    Args:
        frame:      BGR numpy array
        box:        [x1, y1, x2, y2]
        inside:     True = person is in zone
        confidence: YOLO confidence score
        show_conf:  show confidence value on label
        show_foot:  draw yellow dot at foot point
    """
    color = (0, 0, 220) if inside else (0, 200, 0)
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    label        = f"Person {confidence:.2f}" if show_conf else "Person"
    (tw, th), _  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y1     = max(y1 - 22, 0)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, label_y1 + 22), color, -1)
    cv2.putText(frame, label, (x1 + 2, label_y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                lineType=cv2.LINE_AA)

    if show_foot:
        foot = get_foot_point(box)
        cv2.circle(frame, foot, radius=5, color=(0, 255, 255), thickness=-1)


def draw_alert_banner(frame: np.ndarray, count: int) -> None:
    """
    Draw a full-width red alert banner at top of frame in-place.
    Does nothing when count <= 0.

    Args:
        frame: BGR numpy array
        count: number of persons currently in restricted zone
    """
    if count <= 0:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 180), thickness=-1)
    text = f"  ALERT: {count} person(s) in restricted area"
    cv2.putText(frame, text, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                lineType=cv2.LINE_AA)
