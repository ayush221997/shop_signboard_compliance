"""
Agent 2 (verification): extract per-word confidences from Document OCR and flag low-confidence words.
Google Vision provides confidence on each Word and Symbol; we use min(symbol confidences) per word
when available, else word-level confidence.  is_doubtful = confidence < 0.85.
"""

from __future__ import annotations

from typing import Any

# Vision reports confidences in [0, 1] for word/symbol.
DOUBT_THRESHOLD = 0.85


def _min_symbol_confidence(word: Any) -> float | None:
    syms = list(word.symbols)
    if not syms:
        return None
    confs: list[float] = []
    for s in syms:
        if s.confidence is not None:
            confs.append(float(s.confidence))
    if not confs:
        return None
    return min(confs)


def _word_confidence_and_doubtful(word: Any) -> tuple[float, bool]:
    """
    Prefer symbol-level min; fall back to word.confidence; default to 1.0 if missing.
    """
    cmin = _min_symbol_confidence(word)
    if cmin is not None:
        c = cmin
    else:
        wc = getattr(word, "confidence", None)
        c = float(wc) if wc is not None and float(wc) > 0 else 1.0
    c = max(0.0, min(1.0, c))
    return round(c, 3), c < DOUBT_THRESHOLD


def extract_verified_words_from_block(block: Any) -> list[tuple[str, float, bool]]:
    """(text, confidence, is_doubtful) for each word."""
    out: list[tuple[str, float, bool]] = []
    for para in block.paragraphs:
        for word in para.words:
            t = "".join(s.text for s in word.symbols)
            if not t:
                continue
            c, doubtful = _word_confidence_and_doubtful(word)
            out.append((t, c, doubtful))
    return out
