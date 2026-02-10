from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CutterConfig:
    # --- Stage 1: separator-first (bubble-tolerant) ---
    white_thresh: int = 235           # что считать "пустым" (почти белое)
    empty_width_ratio: float = 0.80   # пусто по ширине >= 80% => это разрыв (пузырёк допускается)
    min_gap_height: int = 20          # минимум подряд "пустых" строк, чтобы считать разрывом
    min_scene_height: int = 200       # минимум высоты сцены
    accept_if_scenes_at_least: int = 4  # если separator-first дал >= N сцен — принимаем, иначе fallback

    # --- Stage 2: YOLO clustering ---
    yolo_conf: float = 0.35
    y_gap: int = 240                  # разрыв между кластерами персонажей по центрам Y (меньше => больше сцен)
    pad_top: int = 120
    pad_bottom: int = 140
    min_yolo_scene_height: int = 220

    # --- Stage 3: post merge / cleanup ---
    merge_if_gap_small: int = 25      # если между сценами маленький разрыв — склеить
    merge_small_scene_height: int = 160  # слишком маленькие куски склеиваем с соседями


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def is_full_height_panel(img: np.ndarray,
                         min_height_ratio: float = 0.65,
                         content_x_ratio: float = 0.25) -> bool:
    """
    True, если это цельная вертикальная панель:
    - почти на всю высоту
    - контент сконцентрирован в одном вертикальном столбе
    """
    h, w = img.shape[:2]

    # 1) проверка высоты
    if h < 300:  # защита от мусора
        return False

    # 2) контент по яркости
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 235  # не белое

    # по каждой колонке — сколько контента по высоте
    col_density = mask.mean(axis=0)  # (w,)

    # находим непрерывный диапазон колонок с контентом
    content_cols = col_density > 0.15  # колонка "живая"
    if not content_cols.any():
        return False

    # длина максимального непрерывного сегмента по X
    max_run = 0
    run = 0
    for v in content_cols:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    # если есть вертикальный "столб" контента достаточной ширины
    if max_run / w < content_x_ratio:
        return False

    # 3) контент тянется по большей части высоты
    row_density = mask.mean(axis=1)
    if (row_density > 0.1).mean() < min_height_ratio:
        return False

    return True

def cut_by_separators(img_bgr: np.ndarray, cfg: CutterConfig) -> List[Tuple[int, int]]:
    """
    Режем по горизонтальным "пустым" полосам.
    Bubble-tolerant: достаточно, чтобы пусто было на >= empty_width_ratio ширины.
    """
    if img_bgr is None:
        return []

    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    h, w = gray.shape[:2]
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    empty_mask = (gray >= cfg.white_thresh).astype(np.uint8)
    empty_ratio = empty_mask.mean(axis=1)  # доля "пустоты" по ширине
    is_gap = empty_ratio >= cfg.empty_width_ratio

    scenes: List[Tuple[int, int]] = []
    in_scene = False
    start = 0
    gap = 0

    for y in range(h):
        if not is_gap[y]:
            if not in_scene:
                in_scene = True
                start = y
            gap = 0
        else:
            if in_scene:
                gap += 1
                if gap >= cfg.min_gap_height:
                    end = y - gap
                    if end - start >= cfg.min_scene_height:
                        scenes.append((start, end))
                    in_scene = False

    if in_scene and h - start >= cfg.min_scene_height:
        scenes.append((start, h))

    return scenes


def _yolo_person_boxes_xyxy(long_img_bgr: np.ndarray, yolo_model, conf: float) -> List[Tuple[int, int, int, int]]:
    """
    Для ultralytics.YOLO: results[0].boxes.xyxy и boxes.cls (COCO person=0).
    """
    res = yolo_model(long_img_bgr, conf=conf, verbose=False)
    if not res or res[0].boxes is None:
        return []

    xyxy = res[0].boxes.xyxy
    cls = getattr(res[0].boxes, "cls", None)

    out: List[Tuple[int, int, int, int]] = []
    for i in range(len(xyxy)):
        if cls is not None and int(cls[i].item()) != 0:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def cut_by_yolo_clusters(long_img_bgr: np.ndarray, yolo_model, cfg: CutterConfig) -> List[Tuple[int, int]]:
    """
    Режем длинное полотно по кластерам персонажей по оси Y.
    """
    H, W = long_img_bgr.shape[:2]
    persons = _yolo_person_boxes_xyxy(long_img_bgr, yolo_model, cfg.yolo_conf)
    if not persons:
        return []

    centers = sorted([((y1 + y2) // 2, y1, y2) for (_, y1, _, y2) in persons], key=lambda t: t[0])

    groups: List[List[Tuple[int, int, int]]] = []
    cur = [centers[0]]

    for item in centers[1:]:
        cy, y1, y2 = item
        prev_cy = cur[-1][0]
        if cy - prev_cy >= cfg.y_gap:
            groups.append(cur)
            cur = [item]
        else:
            cur.append(item)
    groups.append(cur)

    scenes: List[Tuple[int, int]] = []
    for g in groups:
        y1 = min(t[1] for t in g) - cfg.pad_top
        y2 = max(t[2] for t in g) + cfg.pad_bottom
        y1 = _clip(y1, 0, H)
        y2 = _clip(y2, 0, H)
        if y2 - y1 >= cfg.min_yolo_scene_height:
            scenes.append((y1, y2))

    scenes.sort()
    return scenes


def _merge_close_and_small(scenes: List[Tuple[int, int]], cfg: CutterConfig, total_h: int) -> List[Tuple[int, int]]:
    """
    Склеиваем сцены, если:
    - они почти соприкасаются
    - или какая-то слишком маленькая (чтобы не плодить мусор)
    """
    if not scenes:
        return []

    scenes = sorted(scenes)
    merged: List[List[int]] = [[scenes[0][0], scenes[0][1]]]

    for y1, y2 in scenes[1:]:
        py1, py2 = merged[-1]
        if y1 <= py2 + cfg.merge_if_gap_small:
            merged[-1][1] = max(py2, y2)
        else:
            merged.append([y1, y2])

    # склеим слишком маленькие куски с соседями
    cleaned: List[List[int]] = []
    for seg in merged:
        y1, y2 = seg
        if (y2 - y1) < cfg.merge_small_scene_height and cleaned:
            cleaned[-1][1] = y2
        else:
            cleaned.append([y1, y2])

    # страховка границ
    out = []
    for y1, y2 in cleaned:
        y1 = _clip(y1, 0, total_h)
        y2 = _clip(y2, 0, total_h)
        if y2 > y1:
            out.append((y1, y2))
    return out


def cut_webtoon_cascade(
    long_img_bgr: np.ndarray,
    yolo_model,
    cfg: CutterConfig = CutterConfig(),
) -> List[np.ndarray]:
    """
    Возвращает список сцен (картинок).
    1) separator-first
    2) если мало — yolo clusters
    3) merge/cleanup
    """
    H = long_img_bgr.shape[0]

    # Stage 1: separator-first
    scenes_xy = cut_by_separators(long_img_bgr, cfg)

    final_scenes = []
    for (y1, y2) in scenes_xy:
        seg = long_img_bgr[y1:y2, :]
        if is_full_height_panel(seg):
            final_scenes.append((y1, y2))
        else:
            final_scenes.append((y1, y2))

    # если сцен мало — fallback на YOLO
    if len(final_scenes) < cfg.accept_if_scenes_at_least:
        scenes_xy = cut_by_yolo_clusters(long_img_bgr, yolo_model, cfg)
    else:
        scenes_xy = final_scenes

            # Stage 2
            yolo_xy = cut_by_yolo_clusters(long_img_bgr, yolo_model, cfg)
            if yolo_xy:
                scenes_xy = yolo_xy

    scenes_xy = _merge_close_and_small(scenes_xy, cfg, total_h=H)

    return [long_img_bgr[y1:y2, :].copy() for (y1, y2) in scenes_xy]
