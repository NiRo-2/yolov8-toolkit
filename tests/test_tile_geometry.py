from tile_yolo_dataset.tile_geometry import (
    clip_box_to_tile,
    iter_tile_windows,
    keep_clipped_box,
    nms_xyxy,
    select_empty_tiles,
    xyxy_to_yolo_line,
    yolo_line_to_xyxy,
)


def test_iter_tile_windows_single_when_small():
    assert iter_tile_windows(800, 600, tile=1024, overlap=0.2) == [(0, 0, 800, 600)]


def test_iter_tile_windows_overlap_grid():
    wins = iter_tile_windows(2000, 1024, tile=1024, overlap=0.2)
    assert wins[0] == (0, 0, 1024, 1024)
    assert wins[-1][2] == 2000
    assert len(wins) >= 2


def test_keep_clipped_box_drops_small_remainder():
    clipped = clip_box_to_tile(0, 0, 100, 100, 90, 0, 190, 100)
    assert clipped is not None
    assert keep_clipped_box(100 * 100, clipped, min_frac=0.2) is False


def test_yolo_coordinate_conversion_round_trips():
    xyxy = yolo_line_to_xyxy([3, 0.5, 0.25, 0.4, 0.2], img_w=200, img_h=100)
    assert xyxy == (3, 60.0, 15.0, 140.0, 35.0)
    assert xyxy_to_yolo_line(*xyxy, tile_w=200, tile_h=100) == "3 0.500000 0.250000 0.400000 0.200000"


def test_select_empty_tiles_caps_fraction():
    chosen = select_empty_tiles(
        labelled_count=90, empty_indices=list(range(50)), empty_frac=0.10, seed=0
    )
    # total output would be 90 + len(chosen); empty share ~= 10% of total
    total = 90 + len(chosen)
    assert len(chosen) / total <= 0.10 + 1e-9
    assert len(chosen) >= 1


def test_nms_xyxy_keeps_higher_conf_same_class():
    dets = [
        {"cls": 0, "conf": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"cls": 0, "conf": 0.5, "x1": 1, "y1": 1, "x2": 11, "y2": 11},
        {"cls": 1, "conf": 0.8, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
    ]
    out = nms_xyxy(dets, iou_thresh=0.5)
    assert len(out) == 2
    assert {(d["cls"], round(d["conf"], 1)) for d in out} == {(0, 0.9), (1, 0.8)}
