# tts_edge.py (v2) - Edge-TTS without SSML, tolerant signature, text cleanup
from __future__ import annotations

import asyncio
import re
from pathlib import Path
import edge_tts


DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
DEFAULT_RATE = "+20%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_VOLUME = "+0%"


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _clean_tts_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # remove urls
    text = _URL_RE.sub("", text)

    # strip html/xml-like tags
    text = _TAG_RE.sub(" ", text)

    # remove backslash noise, keep slashes only if surrounded by letters (and/or)
    text = text.replace("\\", " ")
    text = text.replace(" / ", " ")

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def synthesize_to_file(
    text: str,
    out_path: str | Path,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
    volume: str = DEFAULT_VOLUME,
    **_ignored_kwargs,
) -> str:
    """
    Async synthesis to file. Accepts extra kwargs to avoid breaking if callers pass them.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = _clean_tts_text(text)
    if not text:
        raise ValueError("TTS text is empty after cleanup")

    # NOTE: edge-tts expects text=..., NOT ssml=...
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    await communicate.save(str(out_path))
    return str(out_path)


def synthesize(
    text: str,
    out_path: str | Path,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
    volume: str = DEFAULT_VOLUME,
) -> str:
    return asyncio.run(
        synthesize_to_file(
            text=text,
            out_path=out_path,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
    )
