import fitz  # PyMuPDF (нужен только если используешь PDF путь)
from PIL import Image
import io
import base64
import shutil
import os
from panel_extractor.panel_extractor import PanelExtractor


def generate_image_array_from_pdfs(pdf_files):
    images = []
    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        for page in doc:
            pix = page.get_pixmap()
            img = pix.tobytes("png")
            images.append(img)
        doc.close()
    return images


def scale_image(image_bytes, square_size=512):
    image = Image.open(io.BytesIO(image_bytes))

    target_size = square_size
    original_width, original_height = image.size
    ratio = min(target_size / original_width, target_size / original_height)
    new_width = max(1, int(original_width * ratio))
    new_height = max(1, int(original_height * ratio))

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def scale_base64_image(base64_image, square_size=512):
    image_bytes = base64.b64decode(base64_image)
    scaled_image_bytes = scale_image(image_bytes, square_size)
    return base64.b64encode(scaled_image_bytes).decode("utf-8")


def encode_images_to_base64(image_array):
    return [base64.b64encode(img_bytes).decode("utf-8") for img_bytes in image_array]


def extract_all_pages_as_images(filename):
    image_array = generate_image_array_from_pdfs([filename])
    scaled_images = [scale_image(img) for img in image_array]

    return {
        "scaled": encode_images_to_base64(scaled_images),
        "full": encode_images_to_base64(image_array),
    }


def save_important_pages(volume, profile_pages, chapter_pages, manga, volume_number):
    profile_dir = f"{manga}/v{volume_number}/profiles"
    chapter_dir = f"{manga}/v{volume_number}/chapters"

    if os.path.exists(profile_dir):
        shutil.rmtree(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)

    if os.path.exists(chapter_dir):
        shutil.rmtree(chapter_dir)
    os.makedirs(chapter_dir, exist_ok=True)

    for i in profile_pages:
        if 0 <= i < len(volume):
            with open(f"{profile_dir}/{i}.png", "wb") as f:
                f.write(base64.b64decode(volume[i]))

    for i in chapter_pages:
        if 0 <= i < len(volume):
            with open(f"{chapter_dir}/{i}.png", "wb") as f:
                f.write(base64.b64decode(volume[i]))


def save_all_pages(volume, manga, volume_number):
    pages_dir = f"{manga}/v{volume_number}/pages"

    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir)
    os.makedirs(pages_dir, exist_ok=True)

    for i, img in enumerate(volume):
        with open(f"{pages_dir}/{i}.png", "wb") as f:
            f.write(base64.b64decode(img))

    return pages_dir


def extract_panels(segment):
    """
    segment ожидается dict с ключами:
      segment["images_unscaled"] -> list[str base64]
    Возвращает dict с segment["panels"] (dict/структура PanelExtractor) + important_panels если надо.
    """
    base64_images = segment["images_unscaled"]
    extractor = PanelExtractor()

    all_panels = []
    for b64 in base64_images:
        img_bytes = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        panels = extractor.extract_panels(pil)
        all_panels.append(panels)

    segment["panels"] = all_panels
    return segment


def split_volume_into_parts(volume, volume_unscaled, chapter_pages, num_parts):
    """
    Делит volume на num_parts частей, стараясь резать по chapter_pages.
    Если chapter_pages пустой -> fallback: одна "глава" с 0 страницы, режем равномерно.
    Возвращает:
      {
        "scaled_images": [...],
        "unscaled_images": [...],
        "parts": [(start, end), ...]   # end inclusive
      }
    """

    n = len(volume)
    if n == 0:
        return {"scaled_images": [], "unscaled_images": [], "parts": []}

    # --- sanitize chapter_pages ---
    if not chapter_pages:
        print("[WARN] chapter_pages пустой. Fallback: считаем, что контент начинается с 0.")
        chapter_pages = [0]

    # keep only valid ints in range
    cleaned = []
    for x in chapter_pages:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < n:
            cleaned.append(xi)

    cleaned = sorted(set(cleaned))
    if not cleaned:
        print("[WARN] chapter_pages после фильтра пустой. Fallback: [0].")
        cleaned = [0]

    chapter_pages = cleaned
    start0 = chapter_pages[0]

    # total length of relevant content (indices start0..n-1)
    total_length = n - start0
    if num_parts <= 1 or total_length <= 1:
        return {
            "scaled_images": [volume[start0:n]],
            "unscaled_images": [volume_unscaled[start0:n]],
            "parts": [(start0, n - 1)],
        }

    # target cut positions (in absolute page indices)
    # for i=1..num_parts-1 compute ideal boundary
    targets = []
    for i in range(1, num_parts):
        # ideal absolute index
        ideal = start0 + int(round(i * total_length / num_parts))
        targets.append(ideal)

    # snap each target to nearest chapter page >= target (or fallback to n)
    cut_points = []
    for ideal in targets[:-1]:
        nxt = None
        for cp in chapter_pages:
            if cp >= ideal:
                nxt = cp
                break
        if nxt is None:
            nxt = n  # means "no more chapters"
        cut_points.append(nxt)

    # build parts (end inclusive)
    parts = []
    s = start0
    for cp in cut_points:
        e = min(n - 1, cp - 1)
        if e < s:
            # if snapping produced zero-length, skip it
            continue
        parts.append((s, e))
        s = e + 1

    if s <= n - 1:
        parts.append((s, n - 1))

    # if for some reason nothing produced
    if not parts:
        parts = [(start0, n - 1)]

    # pretty print
    for i, (a, b) in enumerate(parts):
        if i == len(parts) - 1:
            print(f"[SPLIT] {a} -> end ({b})")
        else:
            print(f"[SPLIT] {a} -> {b}")

    scaled_images = [volume[a : b + 1] for (a, b) in parts]
    unscaled_images = [volume_unscaled[a : b + 1] for (a, b) in parts]

    return {"scaled_images": scaled_images, "unscaled_images": unscaled_images, "parts": parts}
