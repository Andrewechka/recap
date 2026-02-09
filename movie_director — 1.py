# movie_director.py (v7.4) - MoviePy 2.1.2 compatible
import asyncio
import base64
import os
from pathlib import Path
from io import BytesIO
import math
import random

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from moviepy import (
    AudioFileClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

from tts_edge import synthesize_to_file

# ====== SETTINGS (v7.4) ======
DEFAULT_EDGE_VOICE = "en-US-AndrewMultilingualNeural"
TTS_RATE = "+15%"
TTS_PITCH = "+0Hz"

AUDIO_SPEED = 1.00

CANVAS_W, CANVAS_H = 1280, 720
FPS = 30

ZOOM_IN_TO = 1.08
ZOOM_OUT_FROM = 1.08

PAN_X_MAX = 30
PAN_Y_MAX = 80
VERTICAL_PAN_Y = 260
VERTICAL_ASPECT_THRESHOLD = 1.35

BLACK_FRAME_MEAN_LUMA_MAX = 35

UPSCALE = 1
USE_BLUR_BG = True
BLUR_RADIUS = 18

TARGET_SEC_PER_IMAGE = 0.95
MIN_IMAGES_PER_SEG = 1
MAX_IMAGES_PER_SEG = 8


# ----------------- helpers -----------------

def _speed_audio_clip(audio_clip: AudioFileClip, factor: float) -> AudioFileClip:
    if not factor or factor == 1.0:
        return audio_clip
    try:
        if hasattr(audio_clip, "with_speed_scaled"):
            return audio_clip.with_speed_scaled(factor)
    except Exception:
        pass
    return audio_clip


def _flatten_panels(panels):
    if not panels:
        return []
    if isinstance(panels, list):
        return panels
    if isinstance(panels, dict):
        flat = []
        for v in panels.values():
            if isinstance(v, list):
                flat.extend(v)
        return flat
    return []


def _coerce_b64(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ("image", "b64", "base64", "data"):
            if k in item and isinstance(item[k], str):
                return item[k]
    return None


def _decode_pil_from_b64(b64: str):
    try:
        image_bytes = base64.b64decode(b64)
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        return pil
    except Exception:
        return None


def _mean_luma(pil_rgb: Image.Image) -> float:
    arr = np.array(pil_rgb.convert("L"), dtype=np.float32)
    return float(arr.mean())


def _is_too_dark(pil_rgb: Image.Image) -> bool:
    return _mean_luma(pil_rgb) <= BLACK_FRAME_MEAN_LUMA_MAX


def _fallback_soft_bg(canvas=(CANVAS_W, CANVAS_H)) -> Image.Image:
    w, h = canvas
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        v = int(75 + (y / max(1, h - 1)) * 45)
        arr[y, :, :] = (v, v, v)
    noise = np.random.randint(0, 10, size=(h, w, 1), dtype=np.uint8)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(6))


def _enhance_manga_pil(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=3))
    return img


def _fit_to_canvas(img: Image.Image, canvas=(CANVAS_W, CANVAS_H)) -> Image.Image:
    cw, ch = canvas
    iw, ih = img.size
    ratio = min(cw / iw, ch / ih)
    nw, nh = int(iw * ratio), int(ih * ratio)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _compose_blur_bg(img: Image.Image, canvas=(CANVAS_W, CANVAS_H), blur_radius=BLUR_RADIUS) -> Image.Image:
    cw, ch = canvas
    bg = img.resize((cw, ch), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(blur_radius))
    fg = _fit_to_canvas(img, canvas=canvas)
    out = bg.copy()
    x = (cw - fg.size[0]) // 2
    y = (ch - fg.size[1]) // 2
    out.paste(fg, (x, y))
    return out


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _frames_for_panzoom(canvas_img: Image.Image, duration: float, mode: str, vertical: bool):
    """
    v7.6 motion: subpixel smooth pan/zoom через affine transform + overscan.
    Ключ: масштаб никогда не падает ниже PAN_OVERSCAN (например 1.10),
    а dx/dy всегда ограничены доступным запасом => НИКАКОЙ черноты по краям.
    """
    duration = max(0.05, float(duration))
    base = canvas_img.convert("RGB")
    w, h = base.size

    steps = max(2, int(round(duration * FPS)))

    # --- Overscan / zoom policy ---
    PAN_OVERSCAN = 1.10      # "кадр 110%" чтобы можно было панорамить без вылетов
    ZOOM_IN_EXTRA = 0.04     # лёгкий zoom-in поверх overscan (1.10 -> 1.14)
    SAFETY = 2.0             # запас в пикселях, чтобы гарантированно не ловить край

    # Важно: mode="out" НЕ уменьшает к 1.0, иначе пан начнет вылезать.
    if mode == "in":
        s0, s1 = PAN_OVERSCAN, PAN_OVERSCAN + ZOOM_IN_EXTRA
    else:
        s0, s1 = PAN_OVERSCAN, PAN_OVERSCAN  # "out" превращаем в "держим 110%"

    # --- pan targets (float) ---
    if vertical:
        # по X меньше, по Y больше (top->bottom)
        dx_total = float(random.randint(-max(2, PAN_X_MAX // 2), max(2, PAN_X_MAX // 2)))
        dy_total = float(VERTICAL_PAN_Y)
        direction = 1.0 if random.random() < 0.80 else -1.0
        dy_total *= direction
    else:
        dx_total = float(random.randint(-PAN_X_MAX, PAN_X_MAX))
        dy_total = float(random.randint(-PAN_Y_MAX, PAN_Y_MAX))

    frames = []
    cx = w * 0.5
    cy = h * 0.5

    for i in range(steps):
        t = (i + 1) / steps
        k = _ease_in_out(t)

        scale = s0 + (s1 - s0) * k  # всегда >= 1.10

        # --- clamp pan to available overscan at this scale ---
        max_dx = (w * (scale - 1.0)) / 2.0 - SAFETY
        max_dy = (h * (scale - 1.0)) / 2.0 - SAFETY
        if max_dx < 0:
            max_dx = 0.0
        if max_dy < 0:
            max_dy = 0.0

        dx_target = float(np.clip(dx_total, -max_dx, max_dx))
        dy_target = float(np.clip(dy_total, -max_dy, max_dy))

        dx = dx_target * k
        dy = dy_target * k

        # PIL affine: input = (output - center - shift)/scale + center
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

        frames.append(np.array(frame, dtype=np.uint8))

    return frames




def _video_clip_from_frames(frames, duration: float) -> VideoClip:
    """
    MoviePy 2.1.2:
    VideoClip принимает frame_function ПОЗИЦИОННО (без make_frame=).
    """
    duration = max(0.05, float(duration))
    n = len(frames)
    if n <= 0:
        blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        frames = [blank]
        n = 1

    def frame_function(t):
        idx = int((t / duration) * n)
        if idx >= n:
            idx = n - 1
        return frames[idx]

    clip = VideoClip(frame_function, duration=duration)

    # fps на клипе (не обязательно, но полезно)
    try:
        clip = clip.with_fps(FPS)
    except Exception:
        pass

    return clip


def _pick_pages(segment: dict):
    for k in ("pages", "page_numbers", "page_indices", "page_idxs"):
        v = segment.get(k)
        if isinstance(v, list) and v:
            out = []
            for x in v:
                if isinstance(x, int):
                    out.append(x)
                elif isinstance(x, str) and x.strip().isdigit():
                    out.append(int(x.strip()))
            return out
    return []


def _pick_scene_images(segment: dict):
    important = segment.get("important_panels") or []
    if important:
        return important

    panels = segment.get("panels")
    pages = _pick_pages(segment)

    if pages and isinstance(panels, dict):
        chosen = []
        for p in pages:
            if p in panels and isinstance(panels[p], list):
                chosen.extend(panels[p])
            sp = str(p)
            if sp in panels and isinstance(panels[sp], list):
                chosen.extend(panels[sp])
        if chosen:
            return chosen

    flat = _flatten_panels(panels)
    if flat:
        return flat

    return segment.get("images_unscaled") or []


def _resample_images(images, target_n: int):
    if not images:
        return []
    if target_n <= 1:
        return [images[0]]
    if len(images) <= target_n:
        return images
    idxs = np.linspace(0, len(images) - 1, num=target_n)
    return [images[int(round(x))] for x in idxs]


# ----------------- main API -----------------

async def make_movie(movie_script, manga, volume_number, narration_client=None):
    print("Narrating movie script (Edge-TTS)...")
    await add_narrations_to_script(movie_script, manga, volume_number)
    print("Editing movie together...")
    out_path = create_movie_from_script(movie_script, manga, volume_number)
    print("Movie created successfully!")
    return out_path


async def add_narrations_to_script(script, manga, volume_number):
    out_dir = Path(manga) / f"v{volume_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(3)

    async def one(i, entry):
        async with sem:
            text = (entry.get("text") or "").strip()
            out_path = out_dir / f"temp_audio_{i}.mp3"
            await synthesize_to_file(
                text=text,
                out_path=str(out_path),
                voice=DEFAULT_EDGE_VOICE,
                rate=TTS_RATE,
                pitch=TTS_PITCH,
            )
            entry["narration_path"] = str(out_path)

    await asyncio.gather(*(one(i, e) for i, e in enumerate(script)))


def create_movie_from_script(script, manga, volume_number):
    out_dir = Path(manga) / f"v{volume_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_clips = []
    audio_clips = []

    # global fallback
    global_fallback = None
    for seg in script:
        if not isinstance(seg, dict):
            continue
        candidates = []
        candidates.extend(seg.get("important_panels") or [])
        candidates.extend(_flatten_panels(seg.get("panels")))
        candidates.extend(seg.get("images_unscaled") or [])
        for raw in candidates:
            b64 = _coerce_b64(raw)
            if not b64:
                continue
            pil_src = _decode_pil_from_b64(b64)
            if pil_src is None:
                continue
            pil = _enhance_manga_pil(pil_src)
            if UPSCALE and UPSCALE > 1:
                w, h = pil.size
                pil = pil.resize((w * UPSCALE, h * UPSCALE), Image.Resampling.LANCZOS)

            canvas = _compose_blur_bg(pil, canvas=(CANVAS_W, CANVAS_H), blur_radius=BLUR_RADIUS) if USE_BLUR_BG else _fit_to_canvas(pil, (CANVAS_W, CANVAS_H))
            if _is_too_dark(canvas):
                continue
            global_fallback = canvas
            break
        if global_fallback is not None:
            break

    if global_fallback is None:
        global_fallback = _fallback_soft_bg((CANVAS_W, CANVAS_H))

    last_good_frame = global_fallback
    mode_toggle = 0

    for segment in script:
        if not isinstance(segment, dict):
            continue

        audio_path = segment.get("narration_path")
        if (not audio_path) or (not os.path.exists(audio_path)) or os.path.getsize(audio_path) == 0:
            continue

        audio_clip = AudioFileClip(audio_path)
        audio_clip = _speed_audio_clip(audio_clip, AUDIO_SPEED)
        audio_duration = float(getattr(audio_clip, "duration", 0.0) or 0.0)
        if audio_duration <= 0:
            try:
                audio_clip.close()
            except Exception:
                pass
            continue

        scene_images = _pick_scene_images(segment)

        target_n = int(round(audio_duration / max(0.3, TARGET_SEC_PER_IMAGE)))
        target_n = max(MIN_IMAGES_PER_SEG, min(MAX_IMAGES_PER_SEG, target_n))
        scene_images = _resample_images(scene_images, target_n) if scene_images else []

        n = max(1, len(scene_images)) if scene_images else 1
        base_d = audio_duration / n

        segment_clips = []

        for idx_img, raw in enumerate(scene_images):
            b64 = _coerce_b64(raw)
            if not b64:
                continue

            pil_src = _decode_pil_from_b64(b64)
            if pil_src is None:
                continue

            iw, ih = pil_src.size
            is_vertical = (ih / max(1, iw)) >= VERTICAL_ASPECT_THRESHOLD

            pil = _enhance_manga_pil(pil_src)
            if UPSCALE and UPSCALE > 1:
                w, h = pil.size
                pil = pil.resize((w * UPSCALE, h * UPSCALE), Image.Resampling.LANCZOS)

            canvas = _compose_blur_bg(pil, canvas=(CANVAS_W, CANVAS_H), blur_radius=BLUR_RADIUS) if USE_BLUR_BG else _fit_to_canvas(pil, (CANVAS_W, CANVAS_H))
            if _is_too_dark(canvas):
                continue

            last_good_frame = canvas

            dur = base_d
            if idx_img == len(scene_images) - 1:
                used = base_d * (len(scene_images) - 1)
                dur = max(0.05, audio_duration - used)

            mode = "in" if (mode_toggle % 2 == 0) else "out"
            mode_toggle += 1

            frames = _frames_for_panzoom(canvas, dur, mode, vertical=is_vertical)
            clip = _video_clip_from_frames(frames, dur)
            segment_clips.append(clip)

        if not segment_clips:
            hold = last_good_frame if not _is_too_dark(last_good_frame) else global_fallback
            if hold is None or _is_too_dark(hold):
                hold = _fallback_soft_bg((CANVAS_W, CANVAS_H))

            mode = "in" if (mode_toggle % 2 == 0) else "out"
            mode_toggle += 1

            frames = _frames_for_panzoom(hold, audio_duration, mode, vertical=False)
            segment_video = _video_clip_from_frames(frames, audio_duration)
        else:
            segment_video = concatenate_videoclips(segment_clips, method="chain")
            segment_video = segment_video.with_duration(audio_duration)

        video_clips.append(segment_video)
        audio_clips.append(audio_clip)

    if not video_clips or not audio_clips:
        raise RuntimeError("No clips to assemble (check audio/images).")

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

    for a in audio_clips:
        try:
            a.close()
        except Exception:
            pass

    for k in range(len(script)):
        p = out_dir / f"temp_audio_{k}.mp3"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    return str(out_path)
