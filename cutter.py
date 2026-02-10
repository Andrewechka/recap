import numpy as np
import cv2

class CutterConfig:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    # Additional methods and attributes


def _moving_avg_1d(data, window_size):
    if window_size < 1:
        return []
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def _diag_border_cuts(image):
    # Function implementation
    pass


def _color_change_cuts(image):
    # Function implementation
    pass


def split_scene_adaptive(image):
    # Function implementation
    pass


def is_full_height_panel(image):
    # Function implementation
    pass


def cut_by_separators(image):
    # Function implementation
    pass


def _yolo_person_boxes_xyxy(image):
    # Function implementation
    pass


def cut_by_yolo_clusters(image):
    # Function implementation
    pass


def _merge_close_and_small(boxes):
    # Function implementation
    pass


def cut_webtoon_cascade(image):
    # Function implementation
    pass
