"""
tools/ocr_tool.py  —  EasyOCR wrapper (runs on CPU to preserve GPU VRAM for LLM).

Accepts a raw bytes or base64 image, returns extracted text string.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Union

import numpy as np

logger = logging.getLogger("ocr_tool")


_reader_instance = None

def _get_reader():
    """Lazy-load EasyOCR reader once and cache it."""
    global _reader_instance
    if _reader_instance is not None:
        return _reader_instance
    try:
        import easyocr  # type: ignore
        from config.settings import OCR_LANGUAGES, OCR_GPU
        logger.info(f"Loading EasyOCR (languages={OCR_LANGUAGES}, gpu={OCR_GPU})")
        _reader_instance = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)
        return _reader_instance
    except ImportError:
        raise RuntimeError(
            "EasyOCR not installed. Run: pip install easyocr"
        )


def ocr_bytes(image_bytes: bytes) -> str:
    """
    Run EasyOCR on raw image bytes.
    Returns the extracted text as a single string.
    """
    reader = _get_reader()

    # Convert bytes → numpy array via Pillow (avoids writing temp files)
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}") from e

    try:
        results = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(str(r) for r in results if r)
        logger.info(f"EasyOCR extracted {len(text)} characters")
        return text
    except Exception as e:
        raise RuntimeError(f"EasyOCR inference failed: {e}") from e


def ocr_base64(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """Decode a base64 image string and run OCR on it."""
    try:
        # Strip data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64)
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}") from e
    return ocr_bytes(raw)
