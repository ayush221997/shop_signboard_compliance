"""
Persist uploaded signboard images: local directory (default) or S3-compatible bucket.
"""

from __future__ import annotations

import mimetypes
import os
import uuid
from pathlib import Path
from typing import Optional

# Content-addressed or random names; we use random + extension for simplicity.

_DEFAULT_LOCAL = Path(__file__).resolve().parent / "data" / "uploads"


def _public_base() -> str:
    return (os.getenv("PUBLIC_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")


def _local_dir() -> Path:
    p = os.getenv("LOCAL_STORAGE_DIR", "").strip()
    return Path(p) if p else _DEFAULT_LOCAL


def ensure_local_dir() -> Path:
    d = _local_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ext_for(filename: str | None, content_type: str | None) -> str:
    n = (filename or "").lower()
    for e in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".bmp", ".tiff", ".avif", ".heic"):
        if n.endswith(e):
            return e
    ct = (content_type or "").split(";", 1)[0].lower()
    m = mimetypes.guess_extension(ct)
    if m:
        return m
    return ".bin"


def save_image_bytes(
    data: bytes,
    *,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """
    Store bytes and return a public URL stored in the database.
    """
    ext = _ext_for(filename, content_type)
    key = f"{uuid.uuid4().hex}{ext}"
    bucket = (os.getenv("S3_BUCKET") or "").strip()
    if bucket:
        return _save_s3(data, key, content_type)
    d = ensure_local_dir()
    path = d / key
    path.write_bytes(data)
    return f"{_public_base()}/files/{key}"


def _save_s3(data: bytes, key: str, content_type: Optional[str]) -> str:
    try:
        import boto3  # type: ignore
    except ImportError as e:
        raise RuntimeError("S3_BUCKET is set but boto3 is not installed") from e

    bucket = os.getenv("S3_BUCKET", "").strip()
    region = os.getenv("AWS_REGION", "us-east-1")
    client = boto3.client("s3", region_name=region)
    extra: dict = {}
    if content_type:
        extra["ContentType"] = content_type.split(";", 1)[0]
    client.put_object(Bucket=bucket, Key=key, Body=data, **extra)
    public = os.getenv("S3_PUBLIC_BASE_URL", "").rstrip("/")
    if public:
        return f"{public}/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
