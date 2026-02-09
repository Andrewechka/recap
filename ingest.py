"""
ingest.py
Единый вход для:
- папки с изображениями
- PDF
- одного длинного webtoon изображения

Выход:
List[PIL.Image]  # страницы
"""

from pathlib import Path
from typing import List
from PIL import Image
import os

# -----------------------------
# Настройки
# -----------------------------

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# -----------------------------
# Основные функции
# -----------------------------

def ingest(input_path: str) -> List[Image.Image]:
    """
    Универсальный вход.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if path.is_dir():
        return _load_images_from_folder(path)

    if path.suffix.lower() == ".pdf":
        return _load_images_from_pdf(path)

    if path.suffix.lower() in IMAGE_EXTS:
        # считаем, что это длинный webtoon
        return _split_webtoon_image(path)

    raise ValueError(f"Unsupported input format: {path}")


# -----------------------------
# Реализации
# -----------------------------

def _load_images_from_folder(folder: Path) -> List[Image.Image]:
    images = []
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    )

    if not files:
        raise RuntimeError(f"No images found in folder: {folder}")

    for p in files:
        img = Image.open(p).convert("RGB")
        images.append(img)

    print(f"[INGEST] Loaded {len(images)} images from folder")
    return images


def _load_images_from_pdf(pdf_path: Path) -> List[Image.Image]:
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("pip install pdf2image")

    pages = convert_from_path(pdf_path)
    pages = [p.convert("RGB") for p in pages]

    print(f"[INGEST] Loaded {len(pages)} pages from PDF")
    return pages


def _split_webtoon_image(image_path: Path) -> List[Image.Image]:
    """
    Минимальная версия:
    режем длинную ленту на куски фиксированной высоты.
    (позже сюда можно подставить твой splitter)
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    CHUNK_H = 1400  # безопасная высота "страницы"
    pages = []

    y = 0
    while y < h:
        crop = img.crop((0, y, w, min(y + CHUNK_H, h)))
        pages.append(crop)
        y += CHUNK_H

    print(f"[INGEST] Split webtoon into {len(pages)} chunks")
    return pages
