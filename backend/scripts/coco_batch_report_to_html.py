#!/usr/bin/env python3
"""
Render `coco_batch_ocr_report.json` as a self-contained HTML report:
  aggregate stats + per-image analysis (all rows, optional split sections) + one combined table.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
DEFAULT_IN = BACKEND / "data" / "coco_batch_ocr_report.json"
DEFAULT_OUT = BACKEND / "data" / "coco_batch_ocr_report.html"
DEFAULT_SUMMARY = BACKEND.parent / "docs" / "coco_batch_ocr_report_summary.html"


def e(s: object) -> str:
    return html.escape(str(s) if s is not None else "")


def _per_image_analysis_line(r: dict) -> str:
    """One-line readout for the Analysis column (metrics only — raw OCR string not in JSON)."""
    if not r.get("ok"):
        return f"Vision/geometry run failed: {e(r.get('error'))}"
    parts: list[str] = [
        f"threshed geometry conf {e(r.get('geometry_conf'))}",
        f"agent {e(r.get('compliance_status'))}",
        f"dominant {e(r.get('dom_lang'))}",
    ]
    tlen = r.get("full_text_len")
    if tlen is not None:
        parts.append(f"chars {e(tlen)}")
    if r.get("has_silent_ink"):
        parts.append("possible silent-ink block (OCR miss)")
    g = r.get("gstin_count")
    if g is not None and int(g) > 0:  # type: ignore[arg-type]
        parts.append(f"GSTIN fields ×{e(g)}")
    return " · ".join(parts)


def _per_image_tr_cells(r: dict) -> str:
    err = r.get("error")
    err_cell = f'<span class="err">{e(err)}</span>' if err else "—"
    an = _per_image_analysis_line(r)
    return f"""
  <td>{e(r.get("split"))}</td>
  <td class="path">{e(r.get("relpath"))}</td>
  <td>{"ok" if r.get("ok") else "no"}</td>
  <td>{err_cell}</td>
  <td class="n">{e(r.get("full_text_len"))}</td>
  <td class="n">{e(r.get("vision_blocks"))}</td>
  <td class="n">{e(r.get("returned_text_blocks"))}</td>
  <td class="n">{e(r.get("img_w"))}×{e(r.get("img_h"))}</td>
  <td class="n">{e(r.get("geometry_conf"))}</td>
  <td>{e(r.get("compliance_status"))}</td>
  <td>{e(r.get("dom_lang"))}</td>
  <td class="n">{e(round(float(r.get("duration_s", 0) or 0), 3))}</td>
  <td>{"yes" if r.get("has_silent_ink") else "no"}</td>
  <td class="n">{e(r.get("gstin_count"))}</td>
  <td class="analysis">{e(an)}</td>"""


def build_report_html(data: dict) -> str:
    """Full report: metadata, aggregates, per-image rows (by split + full list), with Analysis column."""
    per_split = data.get("per_split") or {}
    results = [r for r in (data.get("results") or []) if isinstance(r, dict)]
    n = len(results)

    head_rows: list[str] = []
    for sp, agg in per_split.items():
        if not isinstance(agg, dict):
            continue
        head_rows.append(
            f"""<tr>
  <td>{e(sp)}</td>
  <td>{e(agg.get("total"))}</td>
  <td>{e(agg.get("ok"))}</td>
  <td>{e(agg.get("failed"))}</td>
  <td>{e(agg.get("mean_full_text_len"))}</td>
  <td>{e(agg.get("mean_vision_blocks"))}</td>
  <td>{e(agg.get("mean_geometry_confidence"))}</td>
  <td>{e(agg.get("mean_vision_call_duration_s"))}</td>
  <td>{e(agg.get("silent_ink_flags"))}</td>
</tr>"""
        )

    by_sp: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_sp[str(r.get("split") or "?")].append(r)

    split_sections: list[str] = []
    for sp in ("train", "test", "valid"):
        rows = by_sp.get(sp) or []
        if not rows:
            continue
        trs = "".join(f"<tr>{_per_image_tr_cells(x)}</tr>\n" for x in rows)
        split_sections.append(
            f"""<details class="split-details" open>
  <summary><strong>{e(sp)}</strong> — {e(len(rows))} image(s) · per-image analysis</summary>
  <div class="wrap">
  <table>
{_per_image_thead()}    <tbody>
{trs}    </tbody>
  </table>
  </div>
</details>
"""
        )

    all_tbody = "".join(f"<tr>{_per_image_tr_cells(r)}</tr>\n" for r in results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>COCO batch OCR — per-image report</title>
  <style>
    :root {{ --bg: #0f1419; --card: #1a2332; --text: #e6edf3; --muted: #8b949e; --bad: #f85149; --b: #30363d; --sum: #58a6ff; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem; line-height: 1.5; font-size: 14px; max-width: 100%; }}
    h1 {{ font-size: 1.35rem; font-weight: 600; margin: 0 0 0.5rem; }}
    .howto {{ background: #21262d; border: 1px solid var(--b); border-radius: 8px; padding: 0.75rem 1rem; margin: 0 0 1rem; font-size: 13px; color: #c9d1d9; }}
    .howto code {{ background: #161b22; padding: 0.1em 0.35em; border-radius: 4px; font-size: 12px; }}
    .meta {{ color: var(--muted); margin-bottom: 1.25rem; font-size: 13px; white-space: pre-wrap; word-break: break-all; }}
    h2 {{ font-size: 1.05rem; font-weight: 600; margin: 1.75rem 0 0.65rem; color: #c9d1d9; border-bottom: 1px solid var(--b); padding-bottom: 0.35rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 12px; min-width: 64rem; }}
    th, td {{ border: 1px solid var(--b); padding: 0.4rem 0.5rem; text-align: left; vertical-align: top; }}
    th {{ background: var(--card); color: #8b949e; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
    tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}
    .path {{ font-size: 11px; max-width: 20rem; word-break: break-all; font-family: ui-monospace, Consolas, monospace; }}
    .n {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .err {{ color: var(--bad); font-size: 12px; }}
    .wrap {{ overflow-x: auto; border: 1px solid var(--b); border-radius: 6px; max-width: 100%; margin: 0.5rem 0; }}
    .analysis {{ font-size: 11px; color: #8b949e; max-width: 28rem; line-height: 1.4; }}
    .split-details {{ margin: 1rem 0; border: 1px solid var(--b); border-radius: 8px; padding: 0.5rem 0.75rem; background: #161b22; }}
    .split-details summary {{ cursor: pointer; color: var(--sum); font-size: 14px; margin-bottom: 0.5rem; }}
    @media (max-width: 800px) {{ th, td {{ font-size: 11px; padding: 0.3rem; }} }}
  </style>
</head>
<body>
  <h1>Storeboard COCO batch — per-image analysis</h1>
  <div class="howto">
    <strong>Reading this report:</strong> Each image has metrics from the same pipeline as <code>/api/ocr</code> (Vision + threshed geometry + COMPLai agent status).
    The <strong>Analysis</strong> column summarizes the row. Raw line-by-line OCR text is <em>not</em> stored in the JSON export — regenerate the batch with script changes if you need full text.
    Open in <strong>Edge/Chrome</strong> if a preview tab is slow. For OneDrive, use <em>Always keep on this device</em> on the file.
  </div>
  <p class="meta">Zip: {e(data.get("zip"))}
Generated (UTC): {e(data.get("generated_at_utc"))}
State hint: {e(data.get("state_hint"))} · Wall time: {e(data.get("wall_time_s"))} s
Total images: {e(n)}
</p>

  <h2>Planned per split</h2>
  <p class="meta">{e(json.dumps(data.get("per_split_planned") or dict(), indent=2))}</p>

  <h2>Aggregates (train / test / valid)</h2>
  <div class="wrap">
  <table>
    <thead>
      <tr>
        <th>Split</th><th>Total</th><th>OK</th><th>Failed</th>
        <th>Mean text len</th><th>Mean Vision blocks</th>
        <th>Mean geometry conf</th><th>Mean call (s)</th><th>Silent-ink flags</th>
      </tr>
    </thead>
    <tbody>
{"".join(head_rows)}
    </tbody>
  </table>
  </div>

  <h2>Per-image analysis by split (collapsible)</h2>
{"".join(split_sections)}

  <h2>All {e(n)} images (single table · scroll or copy)</h2>
  <div class="wrap">
  <table>
{_per_image_thead()}    <tbody>
{all_tbody}    </tbody>
  </table>
  </div>
  <p class="meta">Source: <code>backend/data/coco_batch_ocr_report.json</code> · Regenerate: <code>python scripts/coco_batch_report_to_html.py</code></p>
</body>
</html>
"""


def _per_image_thead() -> str:
    return """<thead>
      <tr>
        <th>Split</th>
        <th>Path (in zip)</th>
        <th>OK</th>
        <th>Error</th>
        <th>Text len</th>
        <th>Vision blocks</th>
        <th>Returned blocks</th>
        <th>Size W×H</th>
        <th>Geometry conf</th>
        <th>Compliance (agent)</th>
        <th>Dominant lang</th>
        <th>Duration s</th>
        <th>Silent ink</th>
        <th>GSTIN count</th>
        <th>Analysis (per image)</th>
      </tr>
    </thead>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="Input JSON")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output HTML (full per-image + analysis)")
    ap.add_argument(
        "--summary-out",
        default=str(DEFAULT_SUMMARY),
        help="Second copy of the same full report (e.g. docs/ for the repo)",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the second copy to --summary-out",
    )
    ap.add_argument(
        "--desktop-copy",
        action="store_true",
        help="Copy full report to OneDrive/Windows Desktop",
    )
    ap.add_argument(
        "--full-only",
        action="store_true",
        help="Only write --out (no summary path)",
    )
    args = ap.parse_args()
    p = Path(args.in_path)
    if not p.is_file():
        print("Input not found:", p, file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    body = build_report_html(data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print("Full: ", out.resolve())

    bom = "\ufeff"
    if not args.no_summary and not args.full_only:
        s_path = Path(args.summary_out)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(bom + body, encoding="utf-8")
        print("Copy: ", s_path.resolve())

    if args.desktop_copy:
        h = Path.home()
        for desk_dir in (h / "Desktop", h / "OneDrive" / "Desktop"):
            if not desk_dir.is_dir():
                continue
            desk = desk_dir / "coco_batch_ocr_report.html"
            desk.write_text(bom + body, encoding="utf-8")
            print("Desktop:", desk.resolve())
            break
        else:
            print("No Desktop folder found (skipped).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
