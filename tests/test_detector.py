import pytest
from unittest.mock import MagicMock
from detector import PERSON_CLASS_ID, parse_detections



def _make_box(cls_id: int, conf: float, xyxy: tuple) -> MagicMock:
    box = MagicMock()
    box.cls.__getitem__  = MagicMock(return_value=cls_id)
    box.conf.__getitem__ = MagicMock(return_value=conf)
    coord = MagicMock()
    coord.tolist = MagicMock(return_value=list(xyxy))
    box.xyxy.__getitem__ = MagicMock(return_value=coord)
    return box

def _make_results(boxes: list) -> MagicMock:
    results = MagicMock()
    results.boxes = boxes
    return results


class TestParseDetections:
    def test_no_boxes_returns_empty(self):
        assert parse_detections(_make_results([]), 0.5) == []

    def test_person_above_threshold_included(self):
        box  = _make_box(PERSON_CLASS_ID, 0.85, (100, 50, 300, 400))
        dets = parse_detections(_make_results([box]), 0.5)
        assert len(dets) == 1

    def test_person_below_threshold_excluded(self):
        box  = _make_box(PERSON_CLASS_ID, 0.30, (100, 50, 300, 400))
        assert parse_detections(_make_results([box]), 0.5) == []

    def test_non_person_class_excluded(self):
        box  = _make_box(cls_id=2, conf=0.95, xyxy=(0, 0, 100, 200))
        assert parse_detections(_make_results([box]), 0.5) == []

    def test_exact_threshold_included(self):
        box  = _make_box(PERSON_CLASS_ID, 0.50, (0, 0, 100, 200))
        dets = parse_detections(_make_results([box]), 0.50)
        assert len(dets) == 1

    def test_box_coordinates_correct(self):
        box  = _make_box(PERSON_CLASS_ID, 0.80, (10, 20, 300, 400))
        dets = parse_detections(_make_results([box]), 0.5)
        assert dets[0]["box"] == [10, 20, 300, 400]

    def test_confidence_rounded_to_2dp(self):
        box  = _make_box(PERSON_CLASS_ID, 0.876543, (0, 0, 100, 200))
        dets = parse_detections(_make_results([box]), 0.5)
        assert dets[0]["confidence"] == round(0.876543, 2)

    def test_output_has_box_key(self):
        box  = _make_box(PERSON_CLASS_ID, 0.70, (0, 0, 50, 100))
        dets = parse_detections(_make_results([box]), 0.5)
        assert "box" in dets[0]

    def test_output_has_confidence_key(self):
        box  = _make_box(PERSON_CLASS_ID, 0.70, (0, 0, 50, 100))
        dets = parse_detections(_make_results([box]), 0.5)
        assert "confidence" in dets[0]

    def test_box_has_four_coords(self):
        box  = _make_box(PERSON_CLASS_ID, 0.70, (10, 20, 110, 220))
        dets = parse_detections(_make_results([box]), 0.5)
        assert len(dets[0]["box"]) == 4

    def test_coordinates_are_ints(self):
        box  = _make_box(PERSON_CLASS_ID, 0.70, (10.7, 20.3, 110.9, 220.1))
        dets = parse_detections(_make_results([box]), 0.5)
        for coord in dets[0]["box"]:
            assert isinstance(coord, int)

    def test_two_persons_both_returned(self):
        boxes = [
            _make_box(PERSON_CLASS_ID, 0.90, (10, 10, 100, 200)),
            _make_box(PERSON_CLASS_ID, 0.80, (200, 10, 350, 200)),
        ]
        assert len(parse_detections(_make_results(boxes), 0.5)) == 2

    def test_mixed_classes_only_persons(self):
        boxes = [
            _make_box(0, 0.90, (10,  10, 100, 200)),
            _make_box(2, 0.95, (200, 10, 350, 200)),
            _make_box(0, 0.75, (400, 10, 500, 200)),
        ]
        assert len(parse_detections(_make_results(boxes), 0.5)) == 2

    def test_one_above_one_below(self):
        boxes = [
            _make_box(PERSON_CLASS_ID, 0.80, (10, 10, 100, 200)),
            _make_box(PERSON_CLASS_ID, 0.20, (200, 10, 350, 200)),
        ]
        assert len(parse_detections(_make_results(boxes), 0.5)) == 1

    def test_all_below_threshold(self):
        boxes = [
            _make_box(PERSON_CLASS_ID, 0.10, (10, 10, 100, 200)),
            _make_box(PERSON_CLASS_ID, 0.05, (200, 10, 350, 200)),
        ]
        assert parse_detections(_make_results(boxes), 0.5) == []

    def test_high_threshold_filters_all(self):
        box  = _make_box(PERSON_CLASS_ID, 0.85, (0, 0, 100, 200))
        assert parse_detections(_make_results([box]), 0.99) == []

    def test_zero_threshold_keeps_all(self):
        box  = _make_box(PERSON_CLASS_ID, 0.01, (0, 0, 100, 200))
        assert len(parse_detections(_make_results([box]), 0.0)) == 1

    def test_person_class_id_is_zero(self):
        assert PERSON_CLASS_ID == 0
