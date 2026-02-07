import base64
import ipaddress
import os
import socket
from typing import Tuple
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from app.core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

OPENAI_OCR_MODEL_ENV = "OPENAI_OCR_MODEL"
DEFAULT_OCR_MODEL = "gpt-4o-mini"
OCR_DOWNLOAD_TIMEOUT_ENV = "OCR_DOWNLOAD_TIMEOUT_SECONDS"
OCR_MAX_IMAGE_BYTES_ENV = "OCR_MAX_IMAGE_BYTES"
DEFAULT_OCR_DOWNLOAD_TIMEOUT_SECONDS = 10.0
DEFAULT_OCR_MAX_IMAGE_BYTES = 5_000_000

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
}
IMAGE_EXTENSION_TO_TYPE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

_OCR_CLIENT: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _OCR_CLIENT
    if _OCR_CLIENT is None:
        _OCR_CLIENT = OpenAI()
    return _OCR_CLIENT


def _get_download_timeout_seconds() -> float:
    raw = os.getenv(OCR_DOWNLOAD_TIMEOUT_ENV, str(DEFAULT_OCR_DOWNLOAD_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_OCR_DOWNLOAD_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_OCR_DOWNLOAD_TIMEOUT_SECONDS


def _get_max_image_bytes() -> int:
    raw = os.getenv(OCR_MAX_IMAGE_BYTES_ENV, str(DEFAULT_OCR_MAX_IMAGE_BYTES))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_OCR_MAX_IMAGE_BYTES
    return value if value > 0 else DEFAULT_OCR_MAX_IMAGE_BYTES


def _is_private_or_local_host(host: str) -> bool:
    host_lower = host.lower()
    if host_lower in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        address = ipaddress.ip_address(host)
        return (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
        )
    except ValueError:
        pass

    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False

    for info in resolved:
        try:
            address = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
        ):
            return True
    return False


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed for OCR.")
    if not parsed.netloc:
        raise ValueError("Invalid URL.")
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL hostname.")
    if _is_private_or_local_host(host):
        raise ValueError("Private/local network URL is not allowed.")


def _resolve_image_type(content_type: str | None, url: str) -> str:
    if content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        if normalized == "image/jpg":
            normalized = "image/jpeg"
        if normalized in ALLOWED_IMAGE_TYPES:
            return normalized
        if normalized.startswith("image/"):
            raise ValueError("Unsupported image content type.")
        if normalized:
            raise ValueError("URL does not point to an image content type.")

    parsed = urlparse(url)
    path = parsed.path.lower()
    for extension, image_type in IMAGE_EXTENSION_TO_TYPE.items():
        if path.endswith(extension):
            return image_type

    raise ValueError("Unsupported or unknown image content type.")


def _download_image(url: str) -> Tuple[bytes, str]:
    timeout = httpx.Timeout(_get_download_timeout_seconds())
    max_bytes = _get_max_image_bytes()
    headers = {"User-Agent": "AI-Server OCR Fetcher/1.0"}

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers=headers)

    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length:
        parsed_length = None
        try:
            parsed_length = int(content_length)
        except ValueError:
            parsed_length = None
        if parsed_length is not None and parsed_length > max_bytes:
            raise ValueError("Image exceeds max allowed size.")

    image_bytes = response.content
    if len(image_bytes) > max_bytes:
        raise ValueError("Image exceeds max allowed size.")

    content_type = _resolve_image_type(response.headers.get("content-type"), url)
    return image_bytes, content_type


def _extract_text_from_image_bytes(image_bytes: bytes, content_type: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = os.getenv(OPENAI_OCR_MODEL_ENV, DEFAULT_OCR_MODEL)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    image_data_url = f"data:{content_type};base64,{image_base64}"

    client = _get_openai_client()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Extract all readable text from this image. "
                            "Return plain text only. "
                            "Do not summarize or translate."
                        ),
                    },
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        max_output_tokens=1000,
    )
    return response.output_text.strip()


def extract_text_from_image_url(url: str) -> str:
    _validate_url(url)
    image_bytes, content_type = _download_image(url)
    text = _extract_text_from_image_bytes(image_bytes, content_type)
    logger.info("OCR extracted %d chars from %s", len(text), url)
    logger.info("OCR message: %s", text)
    return text
