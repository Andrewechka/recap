# app.py (v1) — stable runner for manga-reader
# Input: folder with images OR PDF
# Output: movie + extracted text

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image

from manga_extraction import (
    extract_all_pages_as_images,
    save_important_pages,
    save_all_pages,
    split_volume_into_parts,
    extract_panels,
    scale_base64_image,
)

from vision_analysis import (
    detect_important_pages,
    get_important_panels,
)

from prompts import (
    DRAMATIC_PROMPT,
    BASIC_PROMPT,
    BASIC_PROMPT_WITH_CONTEXT,
    BASIC_INSTRUCTIONS,
    KEY_PAGE_IDENTIFICATION_INSTRUCTIONS,
    KEY_PANEL_IDENTIFICATION_PROMPT,
    KEY_PANEL_IDENTIFICATION_INSTRUCTIONS,
)

from citation_processing import extract_text_and_citations, extract_script
from movie_director import make_movie

load_dotenv()


# -------------------------
# Helpers
# -------------------------

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_images_from_folder(folder: str) -> List[Image.Image]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
    files.sort(key=_natural_key)
    if not files:
        return []

    out: List[Image.Image] = []
    for fn in files:
        p = os.path.join(folder, fn)
        img = Image.open(p).convert("RGB")
        out.append(img)
    return out


def chunk_list(lst, n_chunks: int) -> List[List]:
    n_chunks = max(1, int(n_chunks))
    if not lst:
        return [[] for _ in range(n_chunks)]
    k, m = divmod(len(lst), n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return [c for c in chunks if c]  # remove empties


def write_text_to_file(movie_script, manga: str, volume_number: int):
    output_dir = os.path.join(manga, f"v{volume_number}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extracted_text.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for segment in movie_script:
            f.write(f"Segment Text:\n{segment.get('text','')}\n\n")
            if "citations" in segment:
                f.write(f"Citations: {', '.join(map(str, segment['citations']))}\n\n")
            f.write("-" * 50 + "\n\n")

    print(f"[OK] Extracted text -> {output_file}")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def retry_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RateLimitError as e:
        print("Rate limit reached. Retrying...")
        raise e
    except APIError as e:
        if "rate limit" in str(e).lower():
            print("API error related to rate limit. Retrying...")
            raise RateLimitError(str(e))
        raise e


def analyze_images_with_vision(
    client: OpenAI,
    pages_scaled_base64: List[str],
    prompt: str,
    instructions: str,
    model: str = "gpt-4o",
    detail: str = "low",
    max_tokens: int = 1400,
):
    """
    Minimal replacement for the removed analyze_images_with_gpt4_vision.
    Returns OpenAI response object.
    """
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": detail,
                    },
                }
                for img_b64 in pages_scaled_base64
            ],
        },
    ]
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )


# -------------------------
# Main pipeline
# -------------------------

async def main(input_path: str, manga: str, volume_number: int, text_only: bool = False,
               jobs_count: int = 7, model: str = "gpt-4o"):
    client = OpenAI()

    # 1) Load pages
    print("[INGEST] Loading input:", input_path)
    pages_pil: List[Image.Image] = []

    if input_path.lower().endswith(".pdf"):
        # Original PDF flow
        volume_scaled_and_unscaled = extract_all_pages_as_images(input_path)
        volume_scaled = volume_scaled_and_unscaled["scaled"]     # base64 scaled
        volume_unscaled = volume_scaled_and_unscaled["full"]     # base64 full
        print("[INGEST] PDF pages:", len(volume_scaled))
    else:
        # Folder flow
        pages_pil = load_images_from_folder(input_path)
        if not pages_pil:
            print("❌ No pages loaded from folder.")
            return

        volume_unscaled = [pil_to_base64(p) for p in pages_pil]
        volume_scaled = [scale_base64_image(b) for b in volume_unscaled]
        print("[INGEST] Folder pages:", len(volume_scaled))

    if not volume_scaled:
        print("❌ No pages loaded.")
        return

    # 2) Save all pages for QA (optional but useful)
    try:
        save_all_pages(volume_unscaled, manga, volume_number)
    except Exception as e:
        print("[WARN] save_all_pages failed:", e)

    # 3) Optional: identify profile/chapter pages (can be skipped)
    profile_pages: List[int] = []
    chapter_pages: List[int] = []

    # If references exist, we try. If not, we just skip safely.
    profile_ref_pdf = os.path.join(manga, "profile-reference.pdf")
    chapter_ref_pdf = os.path.join(manga, "chapter-reference.pdf")

    if os.path.exists(profile_ref_pdf) and os.path.exists(chapter_ref_pdf):
        print("[KEYPAGES] References found. Detecting important pages...")
        profile_reference = extract_all_pages_as_images(profile_ref_pdf)["scaled"]
        chapter_reference = extract_all_pages_as_images(chapter_ref_pdf)["scaled"]

        batch_size = 20

        def process_batch(start_idx: int, pages_batch: List[str]):
            resp = retry_api_call(
                detect_important_pages,
                profile_reference,
                chapter_reference,
                pages_batch,
                client,
                KEY_PAGE_IDENTIFICATION_INSTRUCTIONS,
                KEY_PAGE_IDENTIFICATION_INSTRUCTIONS,
            )
            return start_idx, resp

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            futures = []
            for i in range(0, len(volume_scaled), batch_size):
                futures.append(ex.submit(process_batch, i, volume_scaled[i:i + batch_size]))

            for fut in concurrent.futures.as_completed(futures):
                start_idx, resp = fut.result()
                ip = resp.get("parsed_response", []) or []
                for page in ip:
                    if page.get("type") == "profile":
                        profile_pages.append(page["image_index"] + start_idx)
                    elif page.get("type") == "chapter":
                        chapter_pages.append(page["image_index"] + start_idx)

        profile_pages.sort()
        chapter_pages.sort()

        print("[KEYPAGES] profile_pages:", profile_pages)
        print("[KEYPAGES] chapter_pages:", chapter_pages)

        # Save for QA
        try:
            save_important_pages(volume_scaled, profile_pages, chapter_pages, manga, volume_number)
        except Exception as e:
            print("[WARN] save_important_pages failed:", e)
    else:
        print("[KEYPAGES] No reference PDFs found -> skipping key page detection.")

    # 4) Split into jobs (SAFE even if chapter_pages empty)
    # If no chapter_pages -> split evenly
    if chapter_pages:
        jobs_obj = split_volume_into_parts(volume_scaled, volume_unscaled, chapter_pages, jobs_count)
        parts = jobs_obj["parts"]
        jobs_scaled = jobs_obj["scaled_images"]
        jobs_unscaled = jobs_obj["unscaled_images"]
    else:
        # stable fallback split
        parts = chunk_list(list(range(len(volume_scaled))), jobs_count)
        jobs_scaled = [volume_scaled[min(p):max(p)+1] for p in parts]
        jobs_unscaled = [volume_unscaled[min(p):max(p)+1] for p in parts]

    print(f"[JOBS] Created {len(jobs_scaled)} jobs.")

    # 5) Vision -> movie_script
    # NOTE: This is the expensive part (rate limits). We keep it sequential and limit max_tokens.
    recap_context = ""
    movie_script = []

    # character_profiles from detected profile_pages (optional)
    character_profiles = [volume_scaled[i] for i in profile_pages] if profile_pages else []

    for idx, job_pages_scaled in enumerate(jobs_scaled):
        if not job_pages_scaled:
            continue

        # Smaller jobs reduce TPM issues (you can tune this number)
        # If a job is huge, split it further
        max_imgs_per_call = 10
        sub_jobs = [job_pages_scaled[i:i+max_imgs_per_call] for i in range(0, len(job_pages_scaled), max_imgs_per_call)]

        for sub_i, sub_pages in enumerate(sub_jobs):
            if idx == 0 and sub_i == 0:
                prompt = BASIC_PROMPT
            else:
                prompt = (recap_context + "\n-----\n" + BASIC_PROMPT_WITH_CONTEXT) if recap_context else BASIC_PROMPT

            # If you want more drama:
            # prompt = DRAMATIC_PROMPT if not recap_context else recap_context + "\n-----\n" + DRAMATIC_PROMPT

            print(f"[VISION] Job {idx+1}/{len(jobs_scaled)} chunk {sub_i+1}/{len(sub_jobs)} images={len(sub_pages)}")

            # Call vision
            response = retry_api_call(
                analyze_images_with_vision,
                client,
                sub_pages,
                prompt,
                BASIC_INSTRUCTIONS,
                model,
                "low",
                1400,
            )

            text = response.choices[0].message.content or ""
            recap_context = text[-2000:]  # keep tail as context (cheap + stable)

            # Build citations using SAME indices mapping: we need unscaled pages aligned with this chunk
            # We find corresponding unscaled chunk by position inside overall job:
            # easiest: we won't do perfect mapping across all chunks; use current scaled chunk positions only
            # For better mapping later, we can store global indices in parts.
            # Here: approximate mapping by re-scaling the same chunk from unscaled via lookup
            # (works if job was made from the same sequence)
            # We'll reconstruct unscaled chunk by searching base64 in volume_scaled — stable enough for now.
            # But to avoid O(N^2), we just pass empty list if unsure.
            # If citation_processing relies hard on it, you can disable citations later.

            # Best-effort: take same window from job_unscaled if available
            # (only works when split_volume_into_parts used OR fallback split)
            # We locate matching job_unscaled by idx and then slice by subchunk window.
            try:
                # Find job_unscaled by idx
                job_unscaled = jobs_unscaled[idx]
                start = sub_i * max_imgs_per_call
                end = start + len(sub_pages)
                sub_unscaled = job_unscaled[start:end]
            except Exception:
                sub_unscaled = []

            segs = extract_text_and_citations(text, sub_pages, sub_unscaled)
            movie_script.extend(segs)

    if not movie_script:
        print("❌ movie_script is empty (vision returned nothing).")
        return

    # 6) Extract panels for each segment (optional but helps visuals)
    # Your extract_panels should accept movie_script list and fill segment["panels"] etc.
    try:
        print("[PANELS] Extracting panels...")
        extract_panels(movie_script, manga, volume_number)
    except Exception as e:
        print("[WARN] extract_panels failed:", e)

    # 7) Save text
    write_text_to_file(movie_script, manga, volume_number)

    # 8) Make movie (unless text_only)
    if text_only:
        print("[DONE] text_only=True -> skipping video.")
        return

    print("[MOVIE] Building video...")
    # narration_client is not needed (Edge-TTS is inside your pipeline now)
    await make_movie(movie_script, manga, volume_number, narration_client=None)
    print("[DONE] ✅")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to PDF or folder with images")
    p.add_argument("--manga", required=True, help="Project folder name, e.g. polo")
    p.add_argument("--volume-number", required=True, type=int, help="Volume number, e.g. 50")
    p.add_argument("--text-only", action="store_true", help="Skip video creation")
    p.add_argument("--jobs", type=int, default=7, help="How many parts to split the volume into")
    p.add_argument("--model", type=str, default="gpt-4o", help="Vision model name (default: gpt-4o)")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    asyncio.run(main(args.input, args.manga, args.volume_number, args.text_only, args.jobs, args.model))
