"""Engine and session factory."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from db_models import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


def get_database_url() -> str:
    u = (os.getenv("DATABASE_URL") or "").strip()
    if u:
        if u.startswith("postgres://"):
            u = u.replace("postgres://", "postgresql+psycopg2://", 1)
        elif u.startswith("postgresql://") and "+" not in u.split("://", 1)[0]:
            u = u.replace("postgresql://", "postgresql+psycopg2://", 1)
        return u
    return "sqlite:///./ocr_audit.db"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    fac = get_session_factory()
    db = fac()
    try:
        yield db
    finally:
        db.close()


def _ensure_audit_columns(engine: Engine) -> None:
    """Add new columns on existing DBs (SQLite/Postgres) when models gain fields."""
    insp = inspect(engine)
    if not insp.has_table("audit_records"):
        return
    existing = {c["name"] for c in insp.get_columns("audit_records")}
    additions: list[tuple[str, str]] = [
        ("location_state_code", "VARCHAR(8)"),
        ("threshed_dominant_lang", "VARCHAR(16)"),
        ("threshed_l1_code", "VARCHAR(8)"),
        ("threshed_l1_ratio", "DOUBLE PRECISION"),
        ("threshed_l2_code", "VARCHAR(8)"),
        ("threshed_l2_ratio", "DOUBLE PRECISION"),
        ("threshed_l3_code", "VARCHAR(8)"),
        ("threshed_l3_ratio", "DOUBLE PRECISION"),
    ]
    if engine.dialect.name == "sqlite":
        additions = [
            ("location_state_code", "TEXT"),
            ("threshed_dominant_lang", "TEXT"),
            ("threshed_l1_code", "TEXT"),
            ("threshed_l1_ratio", "REAL"),
            ("threshed_l2_code", "TEXT"),
            ("threshed_l2_ratio", "REAL"),
            ("threshed_l3_code", "TEXT"),
            ("threshed_l3_ratio", "REAL"),
        ]
    for col, typ in additions:
        if col in existing:
            continue
        with engine.begin() as conn:
            conn.execute(
                text(f"ALTER TABLE audit_records ADD COLUMN {col} {typ}")
            )
        existing.add(col)


def init_db() -> None:
    eng = get_engine()
    Base.metadata.create_all(bind=eng)
    _ensure_audit_columns(eng)
