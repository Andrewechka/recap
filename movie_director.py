# movie_director.py (v9) - fixes: narration source + silent fallback + always produce clips
from __future__ import annotations

import asyncio
import base64
import math
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from moviepy import (
    AudioClip,
    AudioFileClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

from tts_edge import synthesize_to_file

# ===================== SETTINGS =====================
DEFAULT_EDGE_VOICE = "en-US-AndrewMultilingualNeural"
TTS_RATE = "+20%"
TTS_PITCH = "+0Hz"
TTS_VOLUME = "+0%"

AUDIO_SPEED = 1.00
AUDIO_FPS = 44100

CANVAS_W, CANVAS_H = 1280, 720
FPS = 30

PAN_OVERSCAN = 1.10
ZOOM_IN_EXTRA = 0.04
SAFETY_PX = 2.0

PAN_X_MAX = 30
PAN_Y_MAX = 80

VERTICAL_ASPECT_THRESHOLD = 1.35
VERTICAL_PAN_Y = 260

LONG_PANEL_H_RATIO = 1.35
MIN_LONG_PANEL_DUR = 1.40
MIN_PANEL_SOURCE_W = 690

USE_BLUR_BG = True
BLUR_RADIUS = 18
BLACK_FRAME_MEAN_LUMA_MAX = 35

TARGET_SEC_PER_IMAGE = 0.95
MIN_IMAGES_PER_SEG = 1
MAX_IMAGES_PER_SEG = 8

# If we have no narration text at all, still show something (silent)
DEFAULT_SILENT_SEG_DUR = 2.2


# ===================== HELPERS =====================
def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _mean_luma(pil_rgb: Image.Image) -> float:
    arr = np.asarray(pil_rgb.convert("L"), dtype=np.float32)
    return float(arr.mean())


def _is_too_dark(pil_rgb: Image.Image) -> bool:
    return _mean_luma(pil_rgb) <= BLACK_FRAME_MEAN_LUMA_MAX


def _decode_pil_from_b64(b64: str) -> Image.Image | None:
    try:
        image_bytes = base64.b64decode(b64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _coerce_b64(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("image", "b64", "base64", "data"):
            if k in item and isinstance(item[k], str):
                return item[k]
    return None


def _flatten_panels(panels: Any) -> List[Any]:
    if not panels:
        return []
    if isinstance(panels, list):
        return panels
    if isinstance(panels, dict):
        flat: List[Any] = []
        for v in panels.values():
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        return flat
    return []


def _pick_scene_images(segment: Dict[str, Any]) -> List[str]:
    imgs: List[Any] = []
    imgs.extend(segment.get("important_panels") or [])
    imgs.extend(_flatten_panels(segment.get("panels")))
    imgs.extend(segment.get("images_unscaled") or [])
    imgs.extend(segment.get("images") or [])

    seen = set()
    out: List[str] = []
    for x in imgs:
        b = _coerce_b64(x)
        if not b or b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _resample_images(images: List[str], target_n: int) -> List[str]:
    if not images or target_n <= 0:
        return []
    if len(images) == target_n:
        return images
    if len(images) > target_n:
        idxs = np.linspace(0, len(images) - 1, target_n).round().astype(int)
        return [images[i] for i in idxs.tolist()]
    return images


def _enhance_manga_pil(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    return img


def _resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    if w <= 0:
        return img
    s = target_w / float(w)
    nh = max(1, int(round(h * s)))
    return img.resize((target_w, nh), Image.Resampling.LANCZOS)


def _fit_to_canvas(img: Image.Image, canvas: Tuple[int, int]) -> Image.Image:
    cw, ch = canvas
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGB", (cw, ch), (32, 32, 32))
    ratio = min(cw / iw, ch / ih)
    nw, nh = max(1, int(round(iw * ratio))), max(1, int(round(ih * ratio)))
    out = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return out


def _blur_cover_bg(img: Image.Image, canvas: Tuple[int, int], blur_radius: int) -> Image.Image:
    cw, ch = canvas
    src = img.convert("RGB")
    w, h = src.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (cw, ch), (32, 32, 32))

    scale_cover = max(cw / w, ch / h)
    bw, bh = max(1, int(round(w * scale_cover))), max(1, int(round(h * scale_cover)))
    bg = src.resize((bw, bh), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg = ImageEnhance.Brightness(bg).enhance(0.92)
    bg = bg.crop(((bw - cw) // 2, (bh - ch) // 2, (bw + cw) // 2, (bh + ch) // 2))
    return bg


def _compose_static_canvas(img: Image.Image) -> Image.Image:
    cw, ch = CANVAS_W, CANVAS_H
    src = img.convert("RGB")
    bg = _blur_cover_bg(src, (cw, ch), BLUR_RADIUS) if USE_BLUR_BG else Image.new("RGB", (cw, ch), (16, 16, 16))
    fg = _fit_to_canvas(src, (cw, ch))

    out = bg.copy()
    x = (cw - fg.size[0]) // 2
    y = (ch - fg.size[1]) // 2
    out.paste(fg, (x, y))
    return out


def _speed_audio_clip(audio_clip: AudioFileClip, factor: float) -> AudioFileClip:
    if not factor or factor == 1.0:
        return audio_clip
    try:
        if hasattr(audio_clip, "with_speed_scaled"):
            return audio_clip.with_speed_scaled(factor)
    except Exception:
        pass
    return audio_clip


def _estimate_duration_from_text(text: str) -> float:
    """Если аудио нет — делаем разумную длительность для silent сегмента."""
    t = (text or "").strip()
    if not t:
        return DEFAULT_SILENT_SEG_DUR
    words = len(t.split())
    # ~2.2 слова/сек (комфортный темп), clamp
    dur = words / 2.2
    return float(max(1.6, min(12.0, dur)))


def _make_silence(duration: float) -> AudioClip:
    duration = max(0.05, float(duration))

    def make_audio_frame(t):
        # mono silence
        return 0.0

    return AudioClip(make_audio_frame, duration=duration, fps=AUDIO_FPS)


# ===================== CLIP BUILDERS (NO frame arrays) =====================
def _static_clip(canvas_img: Image.Image, duration: float) -> VideoClip:
    duration = max(0.05, float(duration))
    base = canvas_img.convert("RGB")
    arr = np.asarray(base, dtype=np.uint8)

    def frame_function(t: float) -> np.ndarray:
        return arr

    return VideoClip(frame_function, duration=duration)


def _panzoom_clip(canvas_img: Image.Image, duration: float, mode: str, vertical: bool) -> VideoClip:
    duration = max(0.05, float(duration))
    base = canvas_img.convert("RGB")
    w, h = base.size

    if mode == "in":
        s0, s1 = PAN_OVERSCAN, PAN_OVERSCAN + ZOOM_IN_EXTRA
    else:
        s0, s1 = PAN_OVERSCAN, PAN_OVERSCAN

    if vertical:
        dx_total = float(random.randint(-max(2, PAN_X_MAX // 2), max(2, PAN_X_MAX // 2)))
        dy_total = float(VERTICAL_PAN_Y) * (1.0 if random.random() < 0.80 else -1.0)
    else:
        dx_total = float(random.randint(-PAN_X_MAX, PAN_X_MAX))
        dy_total = float(random.randint(-PAN_Y_MAX, PAN_Y_MAX))

    cx = w * 0.5
    cy = h * 0.5

    def frame_function(t: float) -> np.ndarray:
        tt = 0.0 if duration <= 0 else max(0.0, min(1.0, t / duration))
        k = _ease_in_out(tt)

        scale = s0 + (s1 - s0) * k  # always >= overscan

        max_dx = (w * (scale - 1.0)) / 2.0 - SAFETY_PX
        max_dy = (h * (scale - 1.0)) / 2.0 - SAFETY_PX
        max_dx = max(0.0, max_dx)
        max_dy = max(0.0, max_dy)

        dx_target = float(np.clip(dx_total, -max_dx, max_dx))
        dy_target = float(np.clip(dy_total, -max_dy, max_dy))

        dx = dx_target * k
        dy = dy_target * k

        a = 1.0 / scale
        e = 1.0 / scale
        b = d = 0.0
        c = (-cx - dx) / scale + cx
        f = (-cy - dy) / scale + cy

        frame = base.transform(
            (w, h),
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BICUBIC,
        )
        return np.asarray(frame, dtype=np.uint8)

    return VideoClip(frame_function, duration=duration)


def _vertical_scroll_clip(panel_rgb: Image.Image, duration: float) -> VideoClip:
    duration = max(0.05, float(duration))
    cw, ch = CANVAS_W, CANVAS_H

    src = panel_rgb.convert("RGB")
    bg = _blur_cover_bg(src, (cw, ch), BLUR_RADIUS) if USE_BLUR_BG else Image.new("RGB", (cw, ch), (16, 16, 16))

    fg = _resize_to_width(src, cw)
    fg_w, fg_h = fg.size

    if fg_h <= ch:
        canvas = bg.copy()
        y = (ch - fg_h) // 2
        canvas.paste(fg, (0, y))
        return _static_clip(canvas, duration)

    scroll = float(fg_h - ch)
    direction = 1.0  # top->bottom
    if random.random() < 0.10:
        direction = -1.0

    if direction > 0:
        y0, y1 = 0.0, scroll
    else:
        y0, y1 = scroll, 0.0

    def frame_function(t: float) -> np.ndarray:
        tt = 0.0 if duration <= 0 else max(0.0, min(1.0, t / duration))
        k = _ease_in_out(tt)
        y = y0 + (y1 - y0) * k
        crop = fg.crop((0, int(round(y)), cw, int(round(y)) + ch))
        out = bg.copy()
        out.paste(crop, (0, 0))
        return np.asarray(out, dtype=np.uint8)

    return VideoClip(frame_function, duration=duration)


# ===================== NARRATION =====================
def _get_narration_text(entry: Dict[str, Any]) -> str:
    # FIX: allow 'text' (what extract_text_and_citations usually produces)
    for k in ("narration", "summary", "text"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


async def add_narrations_to_script(script: List[Dict[str, Any]], manga: str, volume_number: int) -> None:
    out_dir = Path("output") / manga / f"volume_{volume_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    async def one(i: int, entry: Dict[str, Any]) -> None:
        text = _get_narration_text(entry)

        # if no text, keep silent duration hint
        if not text:
            entry["narration_path"] = None
            entry["_silent_dur"] = _estimate_duration_from_text("")
            return

        out_path = out_dir / f"temp_audio_{i}.mp3"
        try:
            entry["narration_path"] = await synthesize_to_file(
                text=text,
                out_path=str(out_path),
                voice=DEFAULT_EDGE_VOICE,
                rate=TTS_RATE,
                pitch=TTS_PITCH,
                volume=TTS_VOLUME,
            )
        except Exception:
            # If TTS fails, fallback to silence, but keep duration based on text
            entry["narration_path"] = None
            entry["_silent_dur"] = _estimate_duration_from_text(text)

    await asyncio.gather(*(one(i, e) for i, e in enumerate(script)))


# ===================== MOVIE =====================
def create_movie_from_script(script: List[Dict[str, Any]], manga: str, volume_number: int) -> str:
    out_dir = Path("output") / manga / f"volume_{volume_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_clips: List[VideoClip] = []
    audio_clips: List[Any] = []  # AudioFileClip | AudioClip

    last_good_canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (24, 24, 24))
    mode_toggle = 0

    for segment in script:
        # Determine audio
        audio_clip = None
        audio_duration = None

        audio_path = segment.get("narration_path")
        if isinstance(audio_path, str) and os.path.exists(audio_path):
            ac = AudioFileClip(audio_path)
            ac = _speed_audio_clip(ac, AUDIO_SPEED)
            d = float(ac.duration or 0.0)
            if d > 0.05:
                audio_clip = ac
                audio_duration = d
            else:
                try:
                    ac.close()
                except Exception:
                    pass

        # fallback: silence if no audio
        if audio_clip is None:
            text = _get_narration_text(segment)
            dur = float(segment.get("_silent_dur") or _estimate_duration_from_text(text))
            audio_clip = _make_silence(dur)
            audio_duration = dur

        # Now build visuals for this duration
        images = _pick_scene_images(segment)
        if not images:
            # hard fallback: last good
            clip = _panzoom_clip(last_good_canvas, audio_duration, "in", vertical=False)
            video_clips.append(clip)
            audio_clips.append(audio_clip)
            continue

        target_n = int(round(audio_duration / max(0.10, TARGET_SEC_PER_IMAGE)))
        target_n = max(MIN_IMAGES_PER_SEG, min(MAX_IMAGES_PER_SEG, target_n))
        images = _resample_images(images, target_n)

        prepared: List[Dict[str, Any]] = []
        for b64 in images:
            pil = _decode_pil_from_b64(b64)
            if pil is None:
                continue
            pil = _enhance_manga_pil(pil)
            canvas = _compose_static_canvas(pil)
            # IMPORTANT: do not drop everything; only skip if extremely dark
            if _is_too_dark(canvas):
                continue

            is_vertical = (pil.size[1] / max(1, pil.size[0])) >= VERTICAL_ASPECT_THRESHOLD
            is_narrow = pil.size[0] < MIN_PANEL_SOURCE_W
            fg_scaled = _resize_to_width(pil, CANVAS_W)
            long_by_height = fg_scaled.size[1] > int(CANVAS_H * LONG_PANEL_H_RATIO)
            is_long_panel = bool(is_vertical and long_by_height and (not is_narrow))

            if is_long_panel:
                ratio = fg_scaled.size[1] / float(CANVAS_H)
                weight = max(1.6, min(4.0, ratio))
            else:
                weight = 1.0

            prepared.append(
                dict(
                    pil=pil,
                    canvas=canvas,
                    is_vertical=is_vertical,
                    is_long_panel=is_long_panel,
                    weight=weight,
                )
            )
            last_good_canvas = canvas

        if not prepared:
            # fallback: last good for entire segment
            clip = _panzoom_clip(last_good_canvas, audio_duration, "in", vertical=False)
            video_clips.append(clip)
            audio_clips.append(audio_clip)
            continue

        base_d = audio_duration / max(1, len(prepared))
        for it in prepared:
            if it["is_long_panel"] and base_d < MIN_LONG_PANEL_DUR:
                need = MIN_LONG_PANEL_DUR / max(0.05, base_d)
                it["weight"] = max(float(it["weight"]), float(need))

        sum_w = max(1e-6, sum(float(x["weight"]) for x in prepared))

        seg_clips: List[VideoClip] = []
        for it in prepared:
            dur = audio_duration * float(it["weight"]) / sum_w
            dur = max(0.10, float(dur))

            mode = "in" if (mode_toggle % 2 == 0) else "out"
            mode_toggle += 1

            if it["is_long_panel"]:
                clip = _vertical_scroll_clip(it["pil"], dur)
            else:
                clip = _panzoom_clip(it["canvas"], dur, mode=mode, vertical=bool(it["is_vertical"]))

            seg_clips.append(clip)

        segment_video = concatenate_videoclips(seg_clips, method="chain").with_duration(audio_duration)
        video_clips.append(segment_video)
        audio_clips.append(audio_clip)

    if not video_clips or not audio_clips:
        raise RuntimeError("No clips to assemble: check that movie_script has entries and images.")

    final_video = concatenate_videoclips(video_clips, method="chain")
    final_audio = concatenate_audioclips(audio_clips)
    final_video = final_video.with_audio(final_audio)

    out_path = out_dir / "recap.mp4"
    final_video.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        temp_audiofile=str(out_dir / "temp-audio.m4a"),
        remove_temp=True,
    )

    # close file-based audio clips
    for a in audio_clips:
        try:
            if hasattr(a, "close"):
                a.close()
        except Exception:
            pass

    return str(out_path)


async def make_movie(movie_script: List[Dict[str, Any]], manga: str, volume_number: int, narration_client=None) -> str:
    await add_narrations_to_script(movie_script, manga, volume_number)
    return create_movie_from_script(movie_script, manga, volume_number)
# movie_director.py (v10) - watchable: trim white margins, static blur BG, capped vertical scroll speed, cleanup temp audios
from __future__ import annotations

import asyncio
import base64
import math
import os
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from moviepy import (
    AudioClip,
    AudioFileClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

from tts_edge import synthesize_to_file

# ===================== SETTINGS =====================
DEFAULT_EDGE_VOICE = "en-US-AndrewMultilingualNeural"
TTS_RATE = "+20%"
TTS_PITCH = "+0Hz"
TTS_VOLUME = "+0%"

AUDIO_SPEED = 1.00
AUDIO_FPS = 44100

CANVAS_W, CANVAS_H = 1280, 720
FPS = 30

# Motion
PAN_OVERSCAN = 1.10          # FG never below 110% when panning
ZOOM_IN_EXTRA = 0.05         # 1.10 -> 1.15
SAFETY_PX = 2.0

PAN_X_MAX = 26               # subtle
PAN_Y_MAX = 48               # subtle
VERTICAL_ASPECT_THRESHOLD = 1.25

# Long panels
LONG_PANEL_H_RATIO = 1.30
MIN_LONG_PANEL_DUR = 1.35
MIN_PANEL_SOURCE_W = 690

# Vertical scroll speed cap (main “watchability” lever)
SCROLL_PX_PER_SEC = 105.0    # cap speed; lower => slower scroll
SCROLL_MIN_PIX = 60.0        # if less than this, treat as static window

# Visuals
USE_BLUR_BG = True
BLUR_RADIUS = 18
BLACK_FRAME_MEAN_LUMA_MAX = 28  # don't over-filter; too aggressive => black gaps

# Images per segment
TARGET_SEC_PER_IMAGE = 1.05
MIN_IMAGES_PER_SEG = 1
MAX_IMAGES_PER_SEG = 7

# Silence fallback
DEFAULT_SILENT_SEG_DUR = 2.2

# Trim margins (white borders in extracted panels)
TRIM_WHITE_THRESH = 242      # pixels >= thresh considered "white-ish"
TRIM_MIN_CONTENT_RATIO = 0.08
TRIM_PAD = 8                 # keep small padding around content

# =====================================================


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _mean_luma(pil_rgb: Image.Image) -> float:
    arr = np.asarray(pil_rgb.convert("L"), dtype=np.float32)
    return float(arr.mean())


def _is_too_dark(pil_rgb: Image.Image) -> bool:
    return _mean_luma(pil_rgb) <= BLACK_FRAME_MEAN_LUMA_MAX


def _decode_pil_from_b64(b64: str) -> Image.Image | None:
    try:
        image_bytes = base64.b64decode(b64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _coerce_b64(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("image", "b64", "base64", "data"):
            if k in item and isinstance(item[k], str):
                return item[k]
    return None


def _flatten_panels(panels: Any) -> List[Any]:
    if not panels:
        return []
    if isinstance(panels, list):
        return panels
    if isinstance(panels, dict):
        flat: List[Any] = []
        for v in panels.values():
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        return flat
    return []


def _pick_scene_images(segment: Dict[str, Any]) -> List[str]:
    imgs: List[Any] = []
    imgs.extend(segment.get("important_panels") or [])
    imgs.extend(_flatten_panels(segment.get("panels")))
    imgs.extend(segment.get("images_unscaled") or [])
    imgs.extend(segment.get("images") or [])

    seen = set()
    out: List[str] = []
    for x in imgs:
        b = _coerce_b64(x)
        if not b or b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def _resample_images(images: List[str], target_n: int) -> List[str]:
    if not images or target_n <= 0:
        return []
    if len(images) == target_n:
        return images
    if len(images) > target_n:
        idxs = np.linspace(0, len(images) - 1, target_n).round().astype(int)
        return [images[i] for i in idxs.tolist()]
    return images


def _enhance_manga_pil(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.07)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=135, threshold=3))
    return img


def _auto_trim_margins(img: Image.Image) -> Image.Image:
    """
    Cuts big white-ish borders (typical for extracted panels).
    Works on grayscale threshold; keeps padding.
    """
    rgb = img.convert("RGB")
    g = np.asarray(rgb.convert("L"), dtype=np.uint8)
    mask = g < TRIM_WHITE_THRESH  # content = non-white-ish

    if mask.mean() < TRIM_MIN_CONTENT_RATIO:
        return rgb  # too empty / too white => don't crop

    ys, xs = np.where(mask)
    if len(xs) < 10 or len(ys) < 10:
        return rgb

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # pad
    x0 = max(0, x0 - TRIM_PAD)
    y0 = max(0, y0 - TRIM_PAD)
    x1 = min(rgb.size[0] - 1, x1 + TRIM_PAD)
    y1 = min(rgb.size[1] - 1, y1 + TRIM_PAD)

    # sanity
    if (x1 - x0) < rgb.size[0] * 0.25 or (y1 - y0) < rgb.size[1] * 0.25:
        # avoid over-aggressive crop
        return rgb

    return rgb.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


def _resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    w, h = img.size
    if w <= 0:
        return img
    s = target_w / float(w)
    nh = max(1, int(round(h * s)))
    return img.resize((target_w, nh), Image.Resampling.LANCZOS)


def _fit_to_canvas(img: Image.Image, canvas: Tuple[int, int]) -> Image.Image:
    cw, ch = canvas
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGB", (cw, ch), (32, 32, 32))
    ratio = min(cw / iw, ch / ih)
    nw, nh = max(1, int(round(iw * ratio))), max(1, int(round(ih * ratio)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _blur_cover_bg(img: Image.Image, canvas: Tuple[int, int], blur_radius: int) -> Image.Image:
    cw, ch = canvas
    src = img.convert("RGB")
    w, h = src.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (cw, ch), (24, 24, 24))

    scale_cover = max(cw / w, ch / h)
    bw, bh = max(1, int(round(w * scale_cover))), max(1, int(round(h * scale_cover)))
    bg = src.resize((bw, bh), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg = ImageEnhance.Brightness(bg).enhance(0.92)
    bg = ImageEnhance.Contrast(bg).enhance(1.03)
    bg = bg.crop(((bw - cw) // 2, (bh - ch) // 2, (bw + cw) // 2, (bh + ch) // 2))
    return bg


def _estimate_duration_from_text(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return DEFAULT_SILENT_SEG_DUR
    words = len(t.split())
    dur = words / 2.2
    return float(max(1.6, min(12.0, dur)))


def _make_silence(duration: float) -> AudioClip:
    duration = max(0.05, float(duration))

    def make_audio_frame(t):
        return 0.0

    return AudioClip(make_audio_frame, duration=duration, fps=AUDIO_FPS)


def _speed_audio_clip(audio_clip: AudioFileClip, factor: float) -> AudioFileClip:
    if not factor or factor == 1.0:
        return audio_clip
    try:
        if hasattr(audio_clip, "with_speed_scaled"):
            return audio_clip.with_speed_scaled(factor)
    except Exception:
        pass
    return audio_clip


def _get_narration_text(entry: Dict[str, Any]) -> str:
    for k in ("narration", "summary", "text"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _build_bg_and_fg_layers(src_img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image | None, bool, bool]:
    """
    Returns:
      bg (canvas RGB)
      fg_layer (canvas RGB)  - content pasted into canvas coordinates
      fg_mask (L) or None    - alpha mask for fg_layer
      is_vertical
      is_long_panel_candidate (for scroll)
    """
    cw, ch = CANVAS_W, CANVAS_H
    src = src_img.convert("RGB")

    # background
    bg = _blur_cover_bg(src, (cw, ch), BLUR_RADIUS) if USE_BLUR_BG else Image.new("RGB", (cw, ch), (24, 24, 24))

    w, h = src.size
    is_vertical = (h / max(1, w)) >= VERTICAL_ASPECT_THRESHOLD

    # Prepare FG
    # If vertical-ish, prefer width-fit (so it's readable), then take a window (crop) if taller than canvas.
    if is_vertical:
        fg_wfit = _resize_to_width(src, cw)
        fw, fh = fg_wfit.size
        is_long = fh > int(ch * LONG_PANEL_H_RATIO) and (w >= MIN_PANEL_SOURCE_W)

        if fh <= ch:
            # paste centered
            fg_layer = Image.new("RGB", (cw, ch), (0, 0, 0))
            mask = Image.new("L", (cw, ch), 0)
            y = (ch - fh) // 2
            fg_layer.paste(fg_wfit, (0, y))
            mask.paste(Image.new("L", (cw, fh), 255), (0, y))
            return bg, fg_layer, mask, is_vertical, is_long

        # taller than canvas => take a nice window (top-biased)
        # window anchor: 0%..20% (avoid always exactly top)
        anchor = random.uniform(0.0, 0.20)
        y0 = int(round((fh - ch) * anchor))
        window = fg_wfit.crop((0, y0, cw, y0 + ch))
        # full-frame FG (no mask needed)
        return bg, window, None, is_vertical, True

    # Non-vertical: fit into canvas; use mask to keep BG visible around it
    fg = _fit_to_canvas(src, (cw, ch))
    fg_layer = Image.new("RGB", (cw, ch), (0, 0, 0))
    mask = Image.new("L", (cw, ch), 0)

    x = (cw - fg.size[0]) // 2
    y = (ch - fg.size[1]) // 2
    fg_layer.paste(fg, (x, y))
    mask.paste(Image.new("L", fg.size, 255), (x, y))
    return bg, fg_layer, mask, is_vertical, False


def _static_layers_clip(bg: Image.Image, fg_layer: Image.Image, fg_mask: Image.Image | None, duration: float) -> VideoClip:
    duration = max(0.05, float(duration))
    bg = bg.convert("RGB")
    fg_layer = fg_layer.convert("RGB")

    if fg_mask is None:
        out = bg.copy()
        out.paste(fg_layer, (0, 0))
        arr = np.asarray(out, dtype=np.uint8)
    else:
        out = bg.copy()
        out.paste(fg_layer, (0, 0), fg_mask)
        arr = np.asarray(out, dtype=np.uint8)

    def frame_function(t: float) -> np.ndarray:
        return arr

    return VideoClip(frame_function, duration=duration)


def _panzoom_layers_clip(
    bg: Image.Image,
    fg_layer: Image.Image,
    fg_mask: Image.Image | None,
    duration: float,
    mode: str,
    vertical: bool,
) -> VideoClip:
    """
    Pan/Zoom ONLY on FG layer. BG is static blur.
    Prevents the “whole screen drifts + black edges” effect.
    """
    duration = max(0.05, float(duration))
    cw, ch = CANVAS_W, CANVAS_H

    bg = bg.convert("RGB")
    fg = fg_layer.convert("RGB")
    mask = fg_mask

    # zoom curve (never below overscan)
    if mode == "in":
        s0, s1 = PAN_OVERSCAN, PAN_OVERSCAN + ZOOM_IN_EXTRA
    else:
        s0, s1 = PAN_OVERSCAN, PAN_OVERSCAN  # keep constant (no zoom-out to 1.0)

    # pan targets (subtle)
    if vertical:
        dx_total = float(random.randint(-max(2, PAN_X_MAX // 2), max(2, PAN_X_MAX // 2)))
        dy_total = float(random.randint(-PAN_Y_MAX, PAN_Y_MAX))
    else:
        dx_total = float(random.randint(-PAN_X_MAX, PAN_X_MAX))
        dy_total = float(random.randint(-PAN_Y_MAX, PAN_Y_MAX))

    cx = cw * 0.5
    cy = ch * 0.5

    def frame_function(t: float) -> np.ndarray:
        tt = 0.0 if duration <= 0 else max(0.0, min(1.0, t / duration))
        k = _ease_in_out(tt)

        scale = s0 + (s1 - s0) * k

        # clamp pan based on overscan
        max_dx = (cw * (scale - 1.0)) / 2.0 - SAFETY_PX
        max_dy = (ch * (scale - 1.0)) / 2.0 - SAFETY_PX
        max_dx = max(0.0, max_dx)
        max_dy = max(0.0, max_dy)

        dx_target = float(np.clip(dx_total, -max_dx, max_dx))
        dy_target = float(np.clip(dy_total, -max_dy, max_dy))

        dx = dx_target * k
        dy = dy_target * k

        a = 1.0 / scale
        e = 1.0 / scale
        b = d = 0.0
        c = (-cx - dx) / scale + cx
        f = (-cy - dy) / scale + cy

        fg_t = fg.transform(
            (cw, ch),
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BICUBIC,
        )
        if mask is not None:
            mask_t = mask.transform(
                (cw, ch),
                Image.Transform.AFFINE,
                (a, b, c, d, e, f),
                resample=Image.Resampling.BICUBIC,
            )
            out = bg.copy()
            out.paste(fg_t, (0, 0), mask_t)
        else:
            out = bg.copy()
            out.paste(fg_t, (0, 0))
        return np.asarray(out, dtype=np.uint8)

    return VideoClip(frame_function, duration=duration)


def _vertical_scroll_clip(src_panel: Image.Image, duration: float) -> VideoClip:
    """
    Vertical window scroll with speed cap.
    If panel extremely long, we DO NOT try to show it all in short audio time.
    We scroll only a limited distance (px/sec * duration) to keep it readable.
    """
    duration = max(0.05, float(duration))
    cw, ch = CANVAS_W, CANVAS_H

    src = src_panel.convert("RGB")
    bg = _blur_cover_bg(src, (cw, ch), BLUR_RADIUS) if USE_BLUR_BG else Image.new("RGB", (cw, ch), (24, 24, 24))

    fg = _resize_to_width(src, cw)
    fw, fh = fg.size

    if fh <= ch:
        out = bg.copy()
        y = (ch - fh) // 2
        out.paste(fg, (0, y))
        arr = np.asarray(out, dtype=np.uint8)

        def frame_function(t: float) -> np.ndarray:
            return arr

        return VideoClip(frame_function, duration=duration)

    # full scroll distance (if we were to show everything)
    full_scroll = float(fh - ch)

    # cap scroll distance to keep speed readable
    max_scroll = min(full_scroll, SCROLL_PX_PER_SEC * float(duration))
    if max_scroll < SCROLL_MIN_PIX:
        # treat as static window (top-biased / mid / bottom)
        anchors = [0.05, 0.35, 0.70]
        anchor = random.choice(anchors)
        y0 = int(round((fh - ch) * anchor))
        window = fg.crop((0, y0, cw, y0 + ch))
        out = bg.copy()
        out.paste(window, (0, 0))
        arr = np.asarray(out, dtype=np.uint8)

        def frame_function(t: float) -> np.ndarray:
            return arr

        return VideoClip(frame_function, duration=duration)

    # choose a start window in the long panel, then scroll only max_scroll
    # top-biased start to preserve reading order
    start_anchor = random.uniform(0.00, 0.25)
    y_start = float((full_scroll - max_scroll) * start_anchor)
    y_end = y_start + max_scroll

    def frame_function(t: float) -> np.ndarray:
        tt = 0.0 if duration <= 0 else max(0.0, min(1.0, t / duration))
        k = _ease_in_out(tt)
        y = y_start + (y_end - y_start) * k
        window = fg.crop((0, int(round(y)), cw, int(round(y)) + ch))
        out = bg.copy()
        out.paste(window, (0, 0))
        return np.asarray(out, dtype=np.uint8)

    return VideoClip(frame_function, duration=duration)


async def add_narrations_to_script(script: List[Dict[str, Any]], manga: str, volume_number: int) -> None:
    out_dir = Path("output") / manga / f"volume_{volume_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    async def one(i: int, entry: Dict[str, Any]) -> None:
        text = _get_narration_text(entry)
        if not text:
            entry["narration_path"] = None
            entry["_silent_dur"] = _estimate_duration_from_text("")
            return

        out_path = out_dir / f"temp_audio_{i:05d}.mp3"
        try:
            entry["narration_path"] = await synthesize_to_file(
                text=text,
                out_path=str(out_path),
                voice=DEFAULT_EDGE_VOICE,
                rate=TTS_RATE,
                pitch=TTS_PITCH,
                volume=TTS_VOLUME,
            )
        except Exception:
            entry["narration_path"] = None
            entry["_silent_dur"] = _estimate_duration_from_text(text)

    await asyncio.gather(*(one(i, e) for i, e in enumerate(script)))


def create_movie_from_script(script: List[Dict[str, Any]], manga: str, volume_number: int) -> str:
    out_dir = Path("output") / manga / f"volume_{volume_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_clips: List[VideoClip] = []
    audio_clips: List[Any] = []  # AudioFileClip | AudioClip

    # last-good fallback (never black)
    last_good_bg = Image.new("RGB", (CANVAS_W, CANVAS_H), (24, 24, 24))
    last_good_fg = Image.new("RGB", (CANVAS_W, CANVAS_H), (24, 24, 24))

    mode_toggle = 0

    for segment in script:
        # ===== audio
        audio_clip = None
        audio_duration = None

        audio_path = segment.get("narration_path")
        if isinstance(audio_path, str) and os.path.exists(audio_path):
            ac = AudioFileClip(audio_path)
            ac = _speed_audio_clip(ac, AUDIO_SPEED)
            d = float(ac.duration or 0.0)
            if d > 0.05:
                audio_clip = ac
                audio_duration = d
            else:
                try:
                    ac.close()
                except Exception:
                    pass

        if audio_clip is None:
            text = _get_narration_text(segment)
            dur = float(segment.get("_silent_dur") or _estimate_duration_from_text(text))
            audio_clip = _make_silence(dur)
            audio_duration = dur

        # ===== images
        images = _pick_scene_images(segment)
        if not images:
            # fallback visual
            bg = last_good_bg
            fg = last_good_fg
            clip = _panzoom_layers_clip(bg, fg, None, audio_duration, "in", vertical=False)
            video_clips.append(clip)
            audio_clips.append(audio_clip)
            continue

        target_n = int(round(audio_duration / max(0.10, TARGET_SEC_PER_IMAGE)))
        target_n = max(MIN_IMAGES_PER_SEG, min(MAX_IMAGES_PER_SEG, target_n))
        images = _resample_images(images, target_n)

        prepared: List[Dict[str, Any]] = []

        for b64 in images:
            pil = _decode_pil_from_b64(b64)
            if pil is None:
                continue

            # Trim borders first (important!)
            pil = _auto_trim_margins(pil)

            # Enhance after trim
            pil = _enhance_manga_pil(pil)

            # Build layers (static bg + fg layer)
            bg, fg_layer, fg_mask, is_vertical, long_candidate = _build_bg_and_fg_layers(pil)

            # Keep last-good even if image is “boring”
            if not _is_too_dark(bg):
                last_good_bg = bg
            if not _is_too_dark(fg_layer):
                last_good_fg = fg_layer

            # Decide if we should use vertical scroll
            # Only if actually tall AFTER width-fit (scroll clip handles cap)
            # Additionally, avoid scroll for very narrow sources
            w0, h0 = pil.size
            is_narrow = w0 < MIN_PANEL_SOURCE_W

            # We treat as scroll when:
            # - vertical
            # - and long_candidate
            # - and not narrow
            use_scroll = bool(is_vertical and long_candidate and (not is_narrow))

            # weight: allocate more time to scroll panels, but not insane
            if use_scroll:
                # approximate "how tall" in canvas terms
                fg_wfit = _resize_to_width(pil, CANVAS_W)
                ratio = fg_wfit.size[1] / float(CANVAS_H)
                weight = max(1.35, min(3.2, ratio))
            else:
                weight = 1.0

            prepared.append(
                dict(
                    pil=pil,
                    bg=bg,
                    fg=fg_layer,
                    mask=fg_mask,
                    is_vertical=is_vertical,
                    use_scroll=use_scroll,
                    weight=weight,
                )
            )

        if not prepared:
            # fallback visual
            clip = _panzoom_layers_clip(last_good_bg, last_good_fg, None, audio_duration, "in", vertical=False)
            video_clips.append(clip)
            audio_clips.append(audio_clip)
            continue

        # enforce minimum duration for scroll panels (within segment budget)
        base_d = audio_duration / max(1, len(prepared))
        for it in prepared:
            if it["use_scroll"] and base_d < MIN_LONG_PANEL_DUR:
                need = MIN_LONG_PANEL_DUR / max(0.05, base_d)
                it["weight"] = max(float(it["weight"]), float(need))

        sum_w = max(1e-6, sum(float(x["weight"]) for x in prepared))

        seg_clips: List[VideoClip] = []
        for it in prepared:
            dur = audio_duration * float(it["weight"]) / sum_w
            dur = max(0.12, float(dur))

            mode = "in" if (mode_toggle % 2 == 0) else "out"
            mode_toggle += 1

            if it["use_scroll"]:
                clip = _vertical_scroll_clip(it["pil"], dur)
            else:
                clip = _panzoom_layers_clip(it["bg"], it["fg"], it["mask"], dur, mode=mode, vertical=bool(it["is_vertical"]))

            seg_clips.append(clip)

        segment_video = concatenate_videoclips(seg_clips, method="chain").with_duration(audio_duration)
        video_clips.append(segment_video)
        audio_clips.append(audio_clip)

    if not video_clips or not audio_clips:
        raise RuntimeError("No clips to assemble: check movie_script structure and image sources.")

    final_video = concatenate_videoclips(video_clips, method="chain")
    final_audio = concatenate_audioclips(audio_clips)
    final_video = final_video.with_audio(final_audio)

    out_path = out_dir / "recap.mp4"
    temp_audiofile = out_dir / "temp-audio.m4a"

    final_video.write_videofile(
        str(out_path),
        codec="libx264",
        audio_codec="aac",
        fps=FPS,
        temp_audiofile=str(temp_audiofile),
        remove_temp=True,
    )

    # close file-based audio clips
    for a in audio_clips:
        try:
            if hasattr(a, "close"):
                a.close()
        except Exception:
            pass

    # ===== cleanup temp audios created by TTS (important!)
    for p in out_dir.glob("temp_audio_*.mp3"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        if temp_audiofile.exists():
            temp_audiofile.unlink()
    except Exception:
        pass

    return str(out_path)


async def make_movie(movie_script: List[Dict[str, Any]], manga: str, volume_number: int, narration_client=None) -> str:
    await add_narrations_to_script(movie_script, manga, volume_number)
    return create_movie_from_script(movie_script, manga, volume_number)
