from cnn_chinese_hw.client_server.rem_dupes import rem_dupes, fast_rem_dupes
from cnn_chinese_hw.stroke_tools.brensenham_line import brensenham_line
from cnn_chinese_hw.stroke_tools.get_vertex import get_vertex
from cnn_chinese_hw.stroke_tools.points_normalized import points_normalized


def test_rem_dupes_preserves_first_occurrence_order():
    values = ["你", "好", "你", "學", "好", "習"]

    assert rem_dupes(values) == ["你", "好", "學", "習"]
    assert sorted(fast_rem_dupes(values)) == ["你", "好", "學", "習"]


def test_points_normalized_scales_to_requested_bounds_and_handles_flat_axis():
    vertical = [[(5, 2), (5, 7)]]
    diagonal = [[(10, 10), (20, 30)]]

    assert points_normalized(vertical, width=10, height=20) == [[(0, 0), (0, 20)]]
    assert points_normalized(diagonal, width=10, height=20) == [[(0, 0), (10, 20)]]


def test_brensenham_line_preserves_requested_direction_and_endpoints():
    assert brensenham_line(0, 0, 3, 0) == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert brensenham_line(3, 0, 0, 0) == [(3, 0), (2, 0), (1, 0), (0, 0)]
    assert brensenham_line(0, 0, 0, 3) == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_get_vertex_collapses_straight_lines_but_preserves_corners():
    assert get_vertex([(0, 0), (250, 0), (500, 0), (1000, 0)]) == [(0, 0), (1000, 0)]
    assert get_vertex([(0, 0), (500, 0), (500, 500)]) == [(0, 0), (500, 0), (500, 500)]


def test_get_vertex_keeps_the_corners_of_a_closed_stroke():
    # first == last gives a zero-length chord: every distance was 0 and the
    # whole loop collapsed to [(0, 0)].
    square = [(0, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)]
    assert get_vertex(square) == square


def test_get_vertex_applies_error_scale_below_the_top_level():
    # (250, 50) deviates 20000 (squared) from the (0,0)-(500,500) chord: above
    # the default threshold (225) but below the scaled one (22500).
    nodes = [(0, 0), (250, 50), (500, 500), (1000, 0)]
    assert get_vertex(nodes) == nodes
    assert get_vertex(nodes, error_scale=100) == [(0, 0), (500, 500), (1000, 0)]
