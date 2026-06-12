"""Shared engine for horizontal YouTube clip batches (topic compilation + outro).

Default: compile related topics into one clip from multiple phrase ranges (gaps removed).
Use --continuous for a single uninterrupted source span (legacy Toni-style).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RENDER = Path(__file__).resolve().parent / "render.py"
APPEND_OUTRO = Path(__file__).resolve().parent / "append_outro.py"

sys.path.insert(0, str(APPEND_OUTRO.parent))
from append_outro import append_outro as composite_outro  # noqa: E402
from audio_corruption import (  # noqa: E402
    filter_clean_ranges,
    load_or_scan_corrupt_ranges,
    range_overlaps_corrupt,
    split_continuous_range,
)

MIN_DURATION = 180.0
PAD_BEFORE = 0.22
PAD_AFTER = 0.10
END_PAD = 0.55
PHRASE_SILENCE = 0.5
SILENCE_CUT = 0.45

FILLERS = {"um", "uh", "umm", "uhh", "hmm", "hm", "ah", "er"}


@dataclass
class ProjectConfig:
    video: Path
    outro: Path
    edit_dir: Path
    source_key: str
    video_duration: float
    clips: list[dict]
    project_name: str = "horizontal clips"


def ensure_ffmpeg_path() -> None:
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if winget.is_dir():
        os.environ["PATH"] = str(winget) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def norm(t: str) -> str:
    return re.sub(r"[^a-z]", "", t.lower())


def is_filler(w: dict) -> bool:
    if w.get("type") != "word":
        return False
    return norm(w.get("text") or "") in FILLERS


def phrase_text(chunk: list[dict]) -> str:
    parts: list[str] = []
    for w in chunk:
        if w.get("type") == "word":
            parts.append((w.get("text") or "").strip())
        elif w.get("type") == "audio_event":
            parts.append(f"({(w.get('text') or '').strip()})")
    return " ".join(p for p in parts if p)


def parse_phrases(words: list[dict]) -> list[dict]:
    phrases: list[dict] = []
    chunk: list[dict] = []
    start: float | None = None

    def flush() -> None:
        nonlocal chunk, start
        if not chunk:
            return
        phrases.append(
            {
                "start": start or 0.0,
                "end": chunk[-1].get("end", start or 0.0),
                "words": list(chunk),
                "text": phrase_text(chunk),
            }
        )
        chunk = []
        start = None

    for w in words:
        if w.get("type") == "spacing":
            s, e = w.get("start"), w.get("end")
            gap = (e - s) if s is not None and e is not None else 0
            if gap >= PHRASE_SILENCE and chunk:
                flush()
            continue
        if w.get("type") in ("word", "audio_event"):
            if start is None:
                start = w.get("start")
            chunk.append(w)
    flush()
    return phrases


def select_phrases(phrases: list[dict], start_anchor: str, end_anchor: str) -> list[dict]:
    start_i = end_i = None
    sa, ea = start_anchor.lower(), end_anchor.lower()
    for i, p in enumerate(phrases):
        if sa in p["text"].lower():
            start_i = i
            break
    if start_i is None:
        raise ValueError(f"start anchor not found: {start_anchor!r}")
    for i in range(start_i, len(phrases)):
        if ea in phrases[i]["text"].lower():
            end_i = i
    if end_i is None:
        raise ValueError(f"end anchor not found: {end_anchor!r}")
    return phrases[start_i : end_i + 1]


def format_quote(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 2
    return text[:half] + " … " + text[-half:]


def word_tokens(text: str) -> list[str]:
    return [t for t in (norm(x) for x in re.findall(r"[a-z0-9']+", text.lower())) if t]


def find_word_span(word_texts: list[str], anchor: str) -> tuple[int, int] | None:
    at = word_tokens(anchor)
    if not at:
        return None
    for i in range(len(word_texts) - len(at) + 1):
        if word_texts[i : i + len(at)] == at:
            return i, i + len(at)
    return None


def phrase_time_at_anchor(phrase: dict, anchor: str, pick: str = "start") -> float:
    words_only = [w for w in phrase["words"] if w.get("type") == "word"]
    texts = [norm((w.get("text") or "")) for w in words_only]
    span = find_word_span(texts, anchor)
    if not span:
        return phrase["start"] if pick == "start" else phrase["end"]
    start_i, end_i = span
    if pick == "start":
        return float(words_only[start_i].get("start", phrase["start"]))
    return float(words_only[end_i - 1].get("end", phrase["end"]))


def trim_words_in_phrase(chunk: list[dict]) -> list[dict]:
    keep: list[dict] = []
    for i, w in enumerate(chunk):
        if w.get("type") != "word":
            keep.append(w)
            continue
        if is_filler(w):
            continue
        raw = (w.get("text") or "").strip()
        if i >= 1 and chunk[i - 1].get("type") == "word":
            prev = (chunk[i - 1].get("text") or "").strip().lower().strip(",")
            cur = raw.lower().strip(",")
            if prev == cur and cur in {"i", "and", "the", "a"}:
                continue
        keep.append(w)
    return keep


def trim_words_to_anchor_span(
    words: list[dict], phrase: dict, start_anchor: str | None, end_anchor: str | None
) -> list[dict]:
    if not start_anchor and not end_anchor:
        return words

    word_idxs = [i for i, w in enumerate(words) if w.get("type") == "word"]
    if not word_idxs:
        return words
    texts = [norm((words[i].get("text") or "")) for i in word_idxs]

    start_wi, end_wi = 0, len(word_idxs)
    if start_anchor:
        span = find_word_span(texts, start_anchor)
        if span:
            start_wi = span[0]
    if end_anchor:
        span = find_word_span(texts, end_anchor)
        if span:
            end_wi = span[1]

    first_i = word_idxs[start_wi]
    last_i = word_idxs[end_wi - 1]
    return words[first_i : last_i + 1]


def build_topic_ranges(
    cfg: ProjectConfig,
    selected: list[dict],
    start_anchor: str,
    end_anchor: str,
    corrupt: list[dict],
) -> tuple[list[dict], int]:
    """One KEEP range per phrase; merge only when gaps are tiny (< SILENCE_CUT)."""
    ranges: list[dict] = []
    last_i = len(selected) - 1
    for i, phrase in enumerate(selected):
        trim_start = start_anchor if i == 0 else None
        trim_end = end_anchor if i == last_i else None
        keep = trim_words_in_phrase(phrase["words"])
        keep = trim_words_to_anchor_span(keep, phrase, trim_start, trim_end)
        if not keep:
            continue
        text = phrase_text(keep)
        word_count = len([w for w in keep if w.get("type") == "word"])
        if word_count < 2 and "(laughs)" not in text:
            continue

        s = max(0.0, keep[0].get("start", phrase["start"]) - PAD_BEFORE)
        e = min(
            cfg.video_duration,
            keep[-1].get("end", phrase["end"]) + (END_PAD if i == last_i else PAD_AFTER),
        )
        if corrupt and range_overlaps_corrupt(s, e, corrupt):
            continue
        ranges.append(
            {
                "source": cfg.source_key,
                "start": round(s, 3),
                "end": round(e, 3),
                "beat": "KEEP",
                "quote": format_quote(text),
                "reason": "Topic phrase — dead air between phrases removed",
            }
        )

    merged: list[dict] = []
    for r in ranges:
        if merged and r["start"] - merged[-1]["end"] < SILENCE_CUT:
            bridge_end = r["end"]
            if corrupt and range_overlaps_corrupt(merged[-1]["end"], bridge_end, corrupt):
                merged.append(dict(r))
            else:
                merged[-1]["end"] = bridge_end
                merged[-1]["quote"] = format_quote(merged[-1]["quote"] + " " + r["quote"])
        else:
            merged.append(dict(r))

    start_p = next((p for p in selected if start_anchor.lower() in p["text"].lower()), None)
    end_p = next((p for p in reversed(selected) if end_anchor.lower() in p["text"].lower()), None)
    if merged and start_p:
        cap_start = max(0.0, phrase_time_at_anchor(start_p, start_anchor, "start") - PAD_BEFORE)
        if merged[0]["start"] < cap_start:
            merged[0]["start"] = round(cap_start, 3)
    if merged and end_p:
        natural_end = min(
            cfg.video_duration,
            end_p["end"] + END_PAD,
        )
        if merged[-1]["end"] < natural_end:
            merged[-1]["end"] = round(natural_end, 3)
    return filter_clean_ranges(merged, corrupt, source_key=cfg.source_key)


def build_continuous_range(
    cfg: ProjectConfig,
    selected: list[dict],
    start_anchor: str,
    end_anchor: str,
    corrupt: list[dict],
) -> list[dict]:
    start_p = selected[0]
    end_p = selected[-1]
    t_start = phrase_time_at_anchor(start_p, start_anchor, "start")
    t_end = phrase_time_at_anchor(end_p, end_anchor, "end")
    s = max(0.0, t_start - PAD_BEFORE)
    e = min(cfg.video_duration, t_end + END_PAD)
    full_text = " ".join(p["text"] for p in selected)
    return split_continuous_range(
        s,
        e,
        corrupt,
        cfg.source_key,
        format_quote(full_text, 500),
        "Continuous take — single span, no internal cuts",
    )


def clip_segments(clip: dict) -> list[dict]:
    """Normalize clip definition to segment list (anchors or wall-clock seconds)."""
    if clip.get("start_seconds") is not None and clip.get("end_seconds") is not None:
        return [
            {
                "start_seconds": float(clip["start_seconds"]),
                "end_seconds": float(clip["end_seconds"]),
            }
        ]
    if clip.get("segments"):
        return clip["segments"]
    if clip.get("start_anchor") and clip.get("end_anchor"):
        return [{"start_anchor": clip["start_anchor"], "end_anchor": clip["end_anchor"]}]
    raise ValueError(
        f"clip {clip.get('id')!r} needs start_seconds/end_seconds, segments, or anchors"
    )


def snap_end_to_phrase(phrases: list[dict], t: float, duration: float) -> float:
    for p in phrases:
        if p["start"] <= t <= p["end"]:
            return min(duration, p["end"] + END_PAD)
    for p in phrases:
        if p["start"] > t and p["start"] - t < 20:
            return min(duration, p["end"] + END_PAD)
    for p in reversed(phrases):
        if p["end"] <= t <= p["end"] + 8:
            return min(duration, p["end"] + END_PAD)
    return min(duration, t + END_PAD)


def snap_start_to_phrase(phrases: list[dict], t: float) -> float:
    for p in phrases:
        if p["start"] <= t <= p["end"]:
            return max(0.0, p["start"] - PAD_BEFORE)
    for p in phrases:
        if p["start"] >= t:
            return max(0.0, p["start"] - PAD_BEFORE)
    return max(0.0, t)


def build_timed_range(
    cfg: ProjectConfig,
    start_s: float,
    end_s: float,
    corrupt: list[dict],
    phrases: list[dict] | None = None,
) -> list[dict]:
    if phrases:
        start_s = snap_start_to_phrase(phrases, start_s)
        end_s = snap_end_to_phrase(phrases, end_s, cfg.video_duration)
    s = max(0.0, start_s)
    e = min(cfg.video_duration, end_s)
    ranges = split_continuous_range(
        s,
        e,
        corrupt,
        cfg.source_key,
        f"{s:.0f}s–{e:.0f}s",
        "Timed segment — continuous take",
    )
    for r in ranges:
        r["stream_copy"] = True
    return ranges


def build_clip_ranges(
    cfg: ProjectConfig,
    phrases: list[dict],
    clip: dict,
    corrupt: list[dict],
    *,
    continuous: bool,
) -> tuple[list[dict], int]:
    """Compile one clip from timed spans and/or anchor-based topic segments."""
    all_ranges: list[dict] = []
    dropped = 0
    use_continuous = continuous or clip.get("continuous")
    for seg in clip_segments(clip):
        if "start_seconds" in seg:
            part = build_timed_range(
                cfg, seg["start_seconds"], seg["end_seconds"], corrupt, phrases=phrases
            )
            all_ranges.extend(part)
            continue
        sa = seg["start_anchor"]
        ea = seg["end_anchor"]
        selected = select_phrases(phrases, sa, ea)
        if use_continuous:
            part = build_continuous_range(cfg, selected, sa, ea, corrupt)
        else:
            part, n = build_topic_ranges(cfg, selected, sa, ea, corrupt)
            dropped += n
        all_ranges.extend(part)
    return all_ranges, dropped


def write_edl(cfg: ProjectConfig, ranges: list[dict], path: Path) -> dict:
    total = sum(r["end"] - r["start"] for r in ranges)
    edl = {
        "version": 1,
        "sources": {cfg.source_key: str(cfg.video.resolve()).replace("\\", "/")},
        "ranges": ranges,
        "grade": "none",
        "overlays": [],
        "subtitles": None,
        "total_duration_s": round(total, 2),
    }
    path.write_text(json.dumps(edl, indent=2), encoding="utf-8")
    return edl


def transcript_path(cfg: ProjectConfig) -> Path:
    return cfg.edit_dir / "transcripts" / f"{cfg.video.stem}.json"


def ensure_transcript_alias(cfg: ProjectConfig) -> None:
    src = transcript_path(cfg)
    alias = cfg.edit_dir / "transcripts" / f"{cfg.source_key}.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() and src.exists():
        alias.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    elif not src.exists() and alias.exists():
        src.write_text(alias.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_transcript_available(cfg: ProjectConfig) -> Path:
    """Return transcript path, creating stub or copying alias when needed."""
    ensure_transcript_alias(cfg)
    tr = transcript_path(cfg)
    if tr.exists():
        return tr
    alias = cfg.edit_dir / "transcripts" / f"{cfg.source_key}.json"
    if alias.exists():
        tr.parent.mkdir(parents=True, exist_ok=True)
        tr.write_text(alias.read_text(encoding="utf-8"), encoding="utf-8")
        return tr
    tr.parent.mkdir(parents=True, exist_ok=True)
    dur = float(cfg.video_duration or 0.0) or 3600.0
    stub = {
        "language_code": "eng",
        "text": "",
        "words": [
            {
                "text": ".",
                "start": 0.0,
                "end": dur,
                "type": "word",
                "speaker_id": "speaker_0",
            }
        ],
        "source": "stub",
    }
    tr.write_text(json.dumps(stub, indent=2), encoding="utf-8")
    ensure_transcript_alias(cfg)
    return tr


def clean_render_work(edl_path: Path) -> None:
    work = edl_path.parent
    for sub in ("clips_graded", "clips_draft"):
        graded = work / sub
        if graded.is_dir():
            for seg in graded.glob("seg_*.mp4"):
                try:
                    seg.unlink()
                except OSError:
                    pass
    for name in ("base.mp4", "base_preview.mp4", "base_draft.mp4", "_concat.txt"):
        p = work / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def render_clip_body(edl_path: Path, body_mp4: Path, *, draft: bool) -> None:
    clean_render_work(edl_path)
    if body_mp4.exists():
        try:
            body_mp4.unlink()
        except OSError:
            pass
    cmd = [
        sys.executable,
        str(RENDER),
        str(edl_path),
        "-o",
        str(body_mp4),
        "--no-subtitles",
        "--no-loudnorm",
    ]
    if draft:
        cmd.append("--draft")
    subprocess.run(cmd, check=True)
    prenorm = body_mp4.with_suffix(".prenorm.mp4")
    if prenorm.exists():
        prenorm.unlink(missing_ok=True)


def render_clip(cfg: ProjectConfig, edl_path: Path, out_mp4: Path, *, draft: bool) -> None:
    body = out_mp4.with_name(out_mp4.stem + "_body.mp4")
    render_clip_body(edl_path, body, draft=draft)
    composite_outro(body, cfg.outro, out_mp4, draft=draft, delete_body=True)


def run(cfg: ProjectConfig, argv: list[str] | None = None) -> None:
    ensure_ffmpeg_path()
    ap = argparse.ArgumentParser(description=f"Build {cfg.project_name}")
    ap.add_argument("--only", nargs="+", metavar="ID")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument(
        "--continuous",
        action="store_true",
        help="Single uninterrupted span per segment (no phrase-level cuts)",
    )
    ap.add_argument(
        "--scan-audio",
        action="store_true",
        help="Run corrupt-audio decode scan (off by default; accents often false-flag)",
    )
    ap.add_argument(
        "--rescan-audio",
        action="store_true",
        help="With --scan-audio, ignore cache and re-scan",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)

    render_mode: str | None = None
    if args.draft:
        render_mode = "draft"
    elif args.final:
        render_mode = "final"
    elif not args.validate_only:
        render_mode = "final"

    clips_dir = cfg.edit_dir / "clips"
    draft_dir = clips_dir / "draft"
    edls_dir = cfg.edit_dir / "edls"
    out_dir = draft_dir if render_mode == "draft" else clips_dir

    transcript = ensure_transcript_available(cfg)
    data = json.loads(transcript.read_text(encoding="utf-8"))
    phrases = parse_phrases(data["words"])
    edls_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scan_audio:
        corrupt = load_or_scan_corrupt_ranges(
            cfg.video,
            cfg.edit_dir,
            cfg.video_duration,
            force_rescan=args.rescan_audio,
        )
    else:
        corrupt = []

    only_ids: set[str] | None = None
    if args.only:
        only_ids = set()
        for token in args.only:
            if token.isdigit():
                prefix = f"{int(token):02d}_"
                only_ids.update(c["id"] for c in cfg.clips if c["id"].startswith(prefix))
            else:
                only_ids.add(token)

    manifest_path = cfg.edit_dir / "clips_manifest.json"
    manifest_by_id: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            for m in json.loads(manifest_path.read_text(encoding="utf-8")):
                manifest_by_id[m["id"]] = m
        except json.JSONDecodeError:
            pass

    failed = False
    for clip in cfg.clips:
        if only_ids and clip["id"] not in only_ids:
            continue

        try:
            ranges, dropped = build_clip_ranges(
                cfg, phrases, clip, corrupt, continuous=args.continuous
            )
        except ValueError as e:
            print(f"FAIL {clip['id']}: {e}", file=sys.stderr)
            failed = True
            continue

        if not ranges:
            print(
                f"SKIP {clip['id']}: no clean audio in anchors "
                f"(corrupt regions excluded)",
                file=sys.stderr,
            )
            failed = True
            continue

        dur = sum(r["end"] - r["start"] for r in ranges)
        ok = dur >= MIN_DURATION
        mode_label = "continuous" if args.continuous else f"{len(ranges)} ranges"
        drop_note = f", {dropped} corrupt phrase(s) skipped" if dropped else ""
        print(
            f"{'OK' if ok else 'SHORT'} {clip['id']}: {dur:.1f}s ({mode_label}{drop_note}) | "
            f"IN: {ranges[0]['quote'][:55]}... | OUT: ...{ranges[-1]['quote'][-55:]}"
        )
        if args.validate_only:
            if not ok:
                failed = True
            continue

        if not ok:
            print(f"WARN {clip['id']}: {dur:.1f}s < {MIN_DURATION}s", file=sys.stderr)
            failed = True
            continue

        edl_path = edls_dir / f"{clip['id']}.json"
        edl = write_edl(cfg, ranges, edl_path)
        out_mp4 = out_dir / f"{clip['title']}.mp4"

        if render_mode is None:
            continue

        if out_mp4.exists() and not args.force:
            print("  (skip — exists)")
        else:
            label = "DRAFT" if render_mode == "draft" else "FINAL"
            print(f"\n=== [{label}] {clip['title']} ===")
            render_clip(cfg, edl_path, out_mp4, draft=(render_mode == "draft"))

        segs = clip_segments(clip)
        if clip.get("start_seconds") is not None:
            sa = f"{clip['start_seconds']:.0f}s"
            ea = f"{clip['end_seconds']:.0f}s"
        else:
            sa = clip.get("start_anchor") or segs[0].get("start_anchor", "")
            ea = clip.get("end_anchor") or segs[-1].get("end_anchor", "")
        manifest_by_id[clip["id"]] = {
            "id": clip["id"],
            "title": clip["title"],
            "file": str(out_mp4.relative_to(cfg.edit_dir)).replace("\\", "/"),
            "duration_s": edl["total_duration_s"],
            "range_count": len(ranges),
            "segment_count": len(segs),
            "cut_mode": "continuous" if args.continuous else "topic_compile",
            "outro_s": 12.4,
            "format": "1280-wide draft + outro" if render_mode == "draft" else "1920x1080 final + outro",
            "segments": segs if len(segs) > 1 else None,
            "start_anchor": sa,
            "end_anchor": ea,
        }

    if args.validate_only:
        sys.exit(1 if failed else 0)

    if render_mode:
        manifest = [manifest_by_id[c["id"]] for c in cfg.clips if c["id"] in manifest_by_id]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nDone — {len(manifest)} {render_mode} clip(s) in {out_dir}")
