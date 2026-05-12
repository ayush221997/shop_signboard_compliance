"""SQLAlchemy models for audit persistence."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

JsonCol = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


class ComplianceStatus(str, enum.Enum):
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


class AuditRecord(Base):
    __tablename__ = "audit_records"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)
    image_url: Mapped[str] = mapped_column(Text, nullable=False)
    raw_ocr_json: Mapped[dict[str, Any]] = mapped_column(JsonCol, nullable=False)
    verified_text_json: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JsonCol, nullable=True
    )
    location_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonCol, nullable=True)
    compliance_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ComplianceStatus.PENDING.value,
    )
    # Queryable compliance / language fields (duplicated from raw_ocr_json.signboard_geometry for SQL reporting)
    location_state_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True, index=True)
    threshed_dominant_lang: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    threshed_l1_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True, index=True)
    threshed_l1_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshed_l2_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    threshed_l2_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshed_l3_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    threshed_l3_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
