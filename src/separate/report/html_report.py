from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _bar(
    segments: list[tuple[float, float]], duration: float, width: int = 80, fill: str = "█"
) -> str:
    if duration <= 0:
        return ""
    line = [" "] * width
    for start, end in segments:
        a = int(start / duration * width)
        b = max(a + 1, int(end / duration * width))
        for i in range(max(0, a), min(width, b)):
            line[i] = fill
    return "".join(line)


def render_html_report(
    *,
    duration: float,
    speakers: list[str],
    regions: list[dict[str, Any]],
    chunks: list[dict[str, Any]] | None = None,
    overlaps: list[dict[str, Any]] | None = None,
    similarities: dict[str, Any] | None = None,
    title: str = "speaker-extract report",
    segments: list[dict[str, Any]] | None = None,
    background: str | None = None,
) -> str:
    chunks = chunks or []
    overlaps = overlaps or []
    similarities = similarities or {}
    segments = segments or []

    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:ui-monospace,Consolas,monospace;background:#111;color:#eee;padding:24px;}",
        "h1,h2{font-family:Georgia,serif;color:#f5f0e8;}",
        ".bar{white-space:pre;background:#1b1b1b;padding:6px 8px;"
        "border-radius:4px;margin:4px 0 12px;}",
        ".meta{color:#aaa;}",
        "table{border-collapse:collapse;margin:12px 0;}",
        "td,th{border:1px solid #333;padding:6px 10px;}",
        ".tag-overlap{color:#ffb454;}",
        ".tag-tse{color:#7bdff2;}",
        ".tag-single{color:#9fe870;}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>Duration: {duration:.2f}s · Speakers: {len(speakers)}"
        + (f" · Background: {html.escape(background)}" if background else "")
        + "</p>",
    ]

    parts.append("<h2>Speaker timelines</h2>")
    for spk in speakers:
        segs = [
            (r["span"]["start"], r["span"]["end"]) for r in regions if r.get("speaker_id") == spk
        ]
        if not segs:
            segs = [(s["start"], s["end"]) for s in segments if s.get("speaker") == spk]
        parts.append(f"<div><strong>{html.escape(spk)}</strong></div>")
        parts.append(f"<div class='bar'>{html.escape(_bar(segs, duration))}</div>")

    ov_segs = [(o["span"]["start"], o["span"]["end"]) for o in overlaps]
    parts.append("<div><strong class='tag-overlap'>Overlap</strong></div>")
    parts.append(f"<div class='bar'>{html.escape(_bar(ov_segs, duration, fill='▓'))}</div>")

    tse_segs = [
        (r["span"]["start"], r["span"]["end"]) for r in regions if r.get("source_type") == "overlap"
    ]
    parts.append("<div><strong class='tag-tse'>TSE invoked</strong></div>")
    parts.append(f"<div class='bar'>{html.escape(_bar(tse_segs, duration, fill='▒'))}</div>")

    if chunks:
        parts.append(
            "<h2>Chunk boundaries</h2><table><tr><th>#</th><th>span</th><th>core</th></tr>"
        )
        for c in chunks:
            parts.append(
                f"<tr><td>{c.get('index')}</td><td>{c.get('span')}</td><td>{c.get('core')}</td></tr>"
            )
        parts.append("</table>")

    parts.append("<h2>Regions</h2><table>")
    parts.append(
        "<tr><th>speaker</th><th>start</th><th>end</th><th>type</th>"
        "<th>match</th><th>diar</th><th>emb</th></tr>"
    )
    for r in regions:
        cls = "tag-overlap" if r.get("source_type") == "overlap" else "tag-single"
        emb = r.get("embedding_similarity")
        emb_s = "" if emb is None else f"{float(emb):.2f}"
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(r.get('speaker_id')))}</td>"
            f"<td>{r['span']['start']:.2f}</td>"
            f"<td>{r['span']['end']:.2f}</td>"
            f"<td class='{cls}'>{html.escape(str(r.get('source_type')))}</td>"
            f"<td>{float(r.get('matching_confidence', 0)):.2f}</td>"
            f"<td>{float(r.get('diarization_confidence', 0)):.2f}</td>"
            f"<td>{emb_s}</td>"
            "</tr>"
        )
    parts.append("</table>")

    if similarities:
        parts.append("<h2>Embedding similarities</h2><pre>")
        parts.append(html.escape(str(similarities)))
        parts.append("</pre>")

    parts.append("</body></html>")
    return "\n".join(parts)


def write_html_report(path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    path.write_text(render_html_report(**kwargs), encoding="utf-8")
    return path
