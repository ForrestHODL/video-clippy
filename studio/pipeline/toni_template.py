"""Generate per-project build_clips.py (continuous blocks, no stream copy)."""
from __future__ import annotations

import json
from pathlib import Path

from studio.paths import helpers_dir, outro_path

TEMPLATE = '''"""Horizontal YouTube clips — {project_name} (continuous takes)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

VIDEO = Path(r"{video}")
OUTRO = Path(r"{outro}")
TRANSCRIPT = Path(__file__).resolve().parent / "transcripts" / "{transcript_stem}.json"
EDIT_DIR = Path(__file__).resolve().parent
CLIPS_DIR = EDIT_DIR / "clips"
DRAFT_DIR = CLIPS_DIR / "draft"
EDLS_DIR = EDIT_DIR / "edls"
RENDER = Path(r"{render}")
APPEND_OUTRO = Path(r"{append_outro}")

sys.path.insert(0, str(APPEND_OUTRO.parent))
from append_outro import append_outro as composite_outro  # noqa: E402

SOURCE = "{source_key}"
VIDEO_DURATION = {duration}
MIN_DURATION = 180.0

CLIPS = {clips_json}


def ensure_env() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def format_quote(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 2
    return text[:half] + " … " + text[-half:]


def build_range(start: float, end: float, quote: str) -> dict:
    return {{
        "source": SOURCE,
        "start": round(max(0.0, start), 3),
        "end": round(min(VIDEO_DURATION, end), 3),
        "beat": "KEEP",
        "quote": format_quote(quote, 500),
        "reason": "Continuous block — no internal cuts",
    }}


def build_clip_ranges(clip: dict) -> list[dict]:
    return [build_range(s["start_seconds"], s["end_seconds"], s.get("quote") or "") for s in clip["segments"]]


def write_edl(ranges: list[dict], path: Path) -> dict:
    total = sum(r["end"] - r["start"] for r in ranges)
    edl = {{
        "version": 1,
        "sources": {{SOURCE: str(VIDEO.resolve()).replace("\\\\", "/")}},
        "ranges": ranges,
        "grade": "none",
        "overlays": [],
        "subtitles": None,
        "total_duration_s": round(total, 2),
    }}
    path.write_text(json.dumps(edl, indent=2), encoding="utf-8")
    return edl


def ensure_transcript_alias() -> None:
    alias = EDIT_DIR / "transcripts" / f"{{SOURCE}}.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() and TRANSCRIPT.exists():
        alias.write_text(TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")


def clean_render_work(edl_path: Path) -> None:
    work = edl_path.parent
    for sub in ("clips_graded", "clips_draft"):
        d = work / sub
        if d.is_dir():
            for seg in d.glob("seg_*.mp4"):
                seg.unlink(missing_ok=True)
    for name in ("base.mp4", "base_preview.mp4", "base_draft.mp4", "_concat.txt"):
        (work / name).unlink(missing_ok=True)


def render_clip(edl_path: Path, out_mp4: Path, *, draft: bool) -> None:
    clean_render_work(edl_path)
    body = out_mp4.with_name(out_mp4.stem + "_body.mp4")
    if body.exists():
        body.unlink()
    cmd = [sys.executable, str(RENDER), str(edl_path), "-o", str(body), "--no-subtitles", "--no-loudnorm"]
    if draft:
        cmd.append("--draft")
    subprocess.run(cmd, check=True)
    composite_outro(body, OUTRO, out_mp4, draft=draft, delete_body=True)


def main() -> None:
    ensure_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = ap.parse_args()
    render_mode = "draft" if args.draft else ("final" if args.final else (None if args.validate_only else "final"))
    out_dir = DRAFT_DIR if render_mode == "draft" else CLIPS_DIR
    ensure_transcript_alias()
    EDLS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    only_ids = None
    if args.only:
        only_ids = set()
        for token in args.only:
            if token.isdigit():
                only_ids.update(c["id"] for c in CLIPS if c["id"].startswith(f"{{int(token):02d}}_"))
            else:
                only_ids.add(token)
    failed = False
    for clip in CLIPS:
        if only_ids and clip["id"] not in only_ids:
            continue
        ranges = build_clip_ranges(clip)
        dur = sum(r["end"] - r["start"] for r in ranges)
        ok = dur >= MIN_DURATION
        print(f"{{'OK' if ok else 'SHORT'}} {{clip['id']}}: {{dur:.1f}}s ({{len(ranges)}} block(s))")
        if args.validate_only:
            failed = failed or not ok
            continue
        if not ok:
            failed = True
            continue
        edl_path = EDLS_DIR / f"{{clip['id']}}.json"
        write_edl(ranges, edl_path)
        out_mp4 = out_dir / f"{{clip['title']}}.mp4"
        if render_mode and (args.force or not out_mp4.exists()):
            print(f"\\n=== [{{render_mode.upper()}}] {{clip['title']}} ===")
            render_clip(edl_path, out_mp4, draft=(render_mode == "draft"))
    if args.validate_only:
        sys.exit(1 if failed else 0)
    if render_mode:
        print(f"\\nDone — {{out_dir}}")


if __name__ == "__main__":
    main()
'''


def write_project(
    edit_dir: Path,
    video: Path,
    source_key: str,
    duration: float,
    clips: list[dict],
    project_name: str,
) -> None:
    body = TEMPLATE.format(
        project_name=project_name,
        video=str(video).replace("\\", "/"),
        outro=str(outro_path()).replace("\\", "/"),
        render=str(helpers_dir() / "render.py").replace("\\", "/"),
        append_outro=str(helpers_dir() / "append_outro.py").replace("\\", "/"),
        transcript_stem=video.stem,
        source_key=source_key,
        duration=duration,
        clips_json=json.dumps(clips, indent=4),
    )
    edit_dir.mkdir(parents=True, exist_ok=True)
    (edit_dir / "build_clips.py").write_text(body, encoding="utf-8")
