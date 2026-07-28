import numpy as np
import pytest
from zone import (
    draw_alert_banner, draw_detection,
    draw_zone, get_foot_point, is_point_in_zone,
)


@pytest.fixture
def square_zone():
    """300×300 square: TL(100,100) TR(400,100) BR(400,400) BL(100,400)."""
    return [(100, 100), (400, 100), (400, 400), (100, 400)]

@pytest.fixture
def triangle_zone():
    return [(200, 50), (500, 400), (50, 400)]

@pytest.fixture
def blank_frame():
    """800×450 black BGR frame."""
    return np.zeros((450, 800, 3), dtype=np.uint8)


class TestGetFootPoint:
    def test_bottom_center_symmetric(self):
        assert get_foot_point([0, 0, 100, 200]) == (50, 200)

    def test_bottom_center_asymmetric(self):
        assert get_foot_point([100, 50, 300, 400]) == (200, 400)

    def test_returns_tuple(self):
        assert isinstance(get_foot_point([0, 0, 10, 10]), tuple)

    def test_returns_ints(self):
        foot = get_foot_point([10, 20, 31, 80])
        assert isinstance(foot[0], int) and isinstance(foot[1], int)

    def test_x_is_center(self):
        assert get_foot_point([100, 0, 200, 300])[0] == 150

    def test_y_is_bottom(self):
        assert get_foot_point([0, 0, 50, 175])[1] == 175

    def test_single_pixel_box(self):
        assert get_foot_point([5, 5, 6, 6]) == (5, 6)

    def test_wide_box(self):
        assert get_foot_point([0, 0, 800, 400]) == (400, 400)

    def test_tall_box(self):
        assert get_foot_point([200, 0, 250, 450])[1] == 450

class TestIsPointInZone:
    def test_center_inside(self, square_zone):
        assert is_point_in_zone((250, 250), square_zone) is True

    def test_far_outside(self, square_zone):
        assert is_point_in_zone((50, 50), square_zone) is False

    def test_top_left_corner(self, square_zone):
        assert is_point_in_zone((100, 100), square_zone) is True

    def test_bottom_right_corner(self, square_zone):
        assert is_point_in_zone((400, 400), square_zone) is True

    def test_just_inside_left_edge(self, square_zone):
        assert is_point_in_zone((101, 250), square_zone) is True

    def test_just_outside_left_edge(self, square_zone):
        assert is_point_in_zone((99, 250), square_zone) is False

    def test_just_inside_top_edge(self, square_zone):
        assert is_point_in_zone((250, 101), square_zone) is True

    def test_just_outside_top_edge(self, square_zone):
        assert is_point_in_zone((250, 99), square_zone) is False

    def test_just_inside_right_edge(self, square_zone):
        assert is_point_in_zone((399, 250), square_zone) is True

    def test_just_outside_right_edge(self, square_zone):
        assert is_point_in_zone((401, 250), square_zone) is False

    def test_negative_coords_outside(self, square_zone):
        assert is_point_in_zone((-1, -1), square_zone) is False

    def test_triangle_centroid_inside(self, triangle_zone):
        assert is_point_in_zone((250, 283), triangle_zone) is True

    def test_triangle_top_right_outside(self, triangle_zone):
        assert is_point_in_zone((490, 60), triangle_zone) is False

    def test_returns_bool_inside(self, square_zone):
        assert isinstance(is_point_in_zone((250, 250), square_zone), bool)

    def test_returns_bool_outside(self, square_zone):
        assert isinstance(is_point_in_zone((0, 0), square_zone), bool)

    def test_person_fully_inside_zone(self, square_zone):
        foot = get_foot_point([150, 150, 250, 350])
        assert is_point_in_zone(foot, square_zone) is True

    def test_person_fully_outside_zone(self, square_zone):
        foot = get_foot_point([500, 10, 600, 90])
        assert is_point_in_zone(foot, square_zone) is False

    def test_box_overlaps_but_feet_outside(self, square_zone):
        foot = get_foot_point([380, 100, 450, 390])   # foot_x = 415 > 400
        assert is_point_in_zone(foot, square_zone) is False


class TestDrawZone:
    def test_does_not_raise(self, blank_frame, square_zone):
        draw_zone(blank_frame, square_zone)

    def test_modifies_frame(self, blank_frame, square_zone):
        before = blank_frame.sum()
        draw_zone(blank_frame, square_zone)
        assert blank_frame.sum() != before

    def test_shape_unchanged(self, blank_frame, square_zone):
        shape = blank_frame.shape
        draw_zone(blank_frame, square_zone)
        assert blank_frame.shape == shape

    def test_triangle_zone(self, blank_frame, triangle_zone):
        draw_zone(blank_frame, triangle_zone)


class TestDrawDetection:
    def test_inside_does_not_raise(self, blank_frame):
        draw_detection(blank_frame, [100, 100, 200, 300], True, 0.85)

    def test_outside_does_not_raise(self, blank_frame):
        draw_detection(blank_frame, [100, 100, 200, 300], False, 0.72)

    def test_modifies_frame(self, blank_frame):
        before = blank_frame.copy()
        draw_detection(blank_frame, [50, 50, 150, 250], False, 0.60)
        assert not np.array_equal(blank_frame, before)

    def test_show_foot(self, blank_frame):
        draw_detection(blank_frame, [200, 100, 350, 400], True, 0.90, show_foot=True)

    def test_hide_confidence(self, blank_frame):
        draw_detection(blank_frame, [200, 100, 350, 400], False, 0.55, show_conf=False)



class TestDrawAlertBanner:
    def test_zero_count_no_change(self, blank_frame):
        before = blank_frame.copy()
        draw_alert_banner(blank_frame, 0)
        assert np.array_equal(blank_frame, before)

    def test_positive_count_modifies_frame(self, blank_frame):
        before = blank_frame.copy()
        draw_alert_banner(blank_frame, 2)
        assert not np.array_equal(blank_frame, before)

    def test_negative_count_no_change(self, blank_frame):
        before = blank_frame.copy()
        draw_alert_banner(blank_frame, -1)
        assert np.array_equal(blank_frame, before)

    def test_large_count_does_not_raise(self, blank_frame):
        draw_alert_banner(blank_frame, 99)
