# video_enhance.py
from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except Exception:
    cv2 = None


def enhance_manga_image(image_bytes: bytes, upscale: int = 2) -> Image.Image:
    """
    Лёгкий улучшайзер для манги:
    - автоконтраст
    - немного шумоподавления (если есть cv2)
    - unsharp mask
    - upscale Lanczos
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # автоконтраст + чуть контраста
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(1.12)

    # шумоподавление (опционально)
    if cv2 is not None:
        arr = np.array(img)
        # мягкий denoise; значения можно подстроить
        arr = cv2.fastNlMeansDenoisingColored(arr, None, 3, 3, 7, 21)
        img = Image.fromarray(arr)

    # резкость / линии
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))

    # upscale
    if upscale and upscale > 1:
        w, h = img.size
        img = img.resize((w * upscale, h * upscale), Image.Resampling.LANCZOS)

    return img


def fit_on_canvas_720p(img: Image.Image, canvas=(1280, 720)) -> Image.Image:
    """
    Вписываем в 1280x720 с чёрным фоном (позже можно blur background).
    """
    bg_w, bg_h = canvas
    iw, ih = img.size
    ratio = min(bg_w / iw, bg_h / ih)
    nw, nh = int(iw * ratio), int(ih * ratio)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    bg = Image.new("RGB", (bg_w, bg_h), (0, 0, 0))
    x = (bg_w - nw) // 2
    y = (bg_h - nh) // 2
    bg.paste(img, (x, y))
    return bg


def make_blurred_background(img: Image.Image, canvas=(1280, 720), blur_radius=18) -> Image.Image:
    """
    Альтернатива чёрному фону: растянуть картинку на фон + blur, поверх — чёткая.
    """
    bg_w, bg_h = canvas
    base = img.copy()

    # фон: растянуть до canvas, потом blur
    bg = base.resize((bg_w, bg_h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(blur_radius))
    fg = fit_on_canvas_720p(img, canvas=canvas)

    # fg уже на чёрном; заменим чёрный на blur через композит:
    # сделаем маску: где чёрный — прозрачный
    fg_arr = np.array(fg)
    mask = (fg_arr.sum(axis=2) > 5).astype(np.uint8) * 255  # не-чёрные
    mask_img = Image.fromarray(mask, mode="L")

    out = Image.composite(fg, bg, mask_img)
    return out
