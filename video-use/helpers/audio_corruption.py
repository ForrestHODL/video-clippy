"""Detect AAC/ffmpeg decode failures and exclude those times from clip EDLs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

COARSE_STEP_S = 8.0
PROBE_DUR_S = 3.0
REFINE_STEP_S = 1.0
EDGE_PAD_S = 0.5


def cache_path(edit_dir: Path) -> Path:
    return edit_dir / "audio_corrupt_ranges.json"


def probe_extract_ok(video: Path, start_s: float, duration_s: float = PROBE_DUR_S) -> bool:
    """Return False when ffmpeg cannot decode audio at this offset (broken AAC, etc.)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "probe.wav"
        # Input seeking (-i then -ss) — accurate decode; -ss before -i false-flags good AAC.
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-ss",
                str(max(0.0, start_s)),
                "-t",
                str(duration_s),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav),
            ],
            capture_output=True,
        )
        return r.returncode == 0 and wav.exists() and wav.stat().st_size > 1000


def _merge_bad_points(bad: list[float], duration: float) -> list[dict]:
    if not bad:
        return []
    bad = sorted(bad)
    ranges: list[dict] = []
    start = bad[0]
    prev = bad[0]
    for t in bad[1:]:
        if t - prev > COARSE_STEP_S * 1.5:
            ranges.append(
                {
                    "start": round(max(0.0, start - EDGE_PAD_S), 3),
                    "end": round(min(duration, prev + PROBE_DUR_S + EDGE_PAD_S), 3),
                }
            )
            start = t
        prev = t
    ranges.append(
        {
            "start": round(max(0.0, start - EDGE_PAD_S), 3),
            "end": round(min(duration, prev + PROBE_DUR_S + EDGE_PAD_S), 3),
        }
    )
    return ranges


def _refine_boundary(video: Path, lo: float, hi: float, want_bad: bool) -> float:
    """Binary-search transition between good and bad decode."""
    while hi - lo > REFINE_STEP_S:
        mid = (lo + hi) / 2
        is_bad = not probe_extract_ok(video, mid, PROBE_DUR_S)
        if is_bad == want_bad:
            lo = mid
        else:
            hi = mid
    return round(hi if want_bad else lo, 3)


def scan_corrupt_ranges(video: Path, duration: float) -> list[dict]:
    """Coarse scan → [{start, end}, ...] of unusable audio (EOF probe failures ignored)."""
    bad: list[float] = []
    # Probes near file end often fail on valid AAC; ignore last ~12s of timeline.
    tail_cutoff = max(0.0, duration - PROBE_DUR_S - COARSE_STEP_S - 2.0)
    t = 0.0
    while t < duration - 1.0:
        if t <= tail_cutoff and not probe_extract_ok(video, t, PROBE_DUR_S):
            bad.append(t)
        t += COARSE_STEP_S

    if not bad:
        return []

    merged = _merge_bad_points(bad, duration)
    refined: list[dict] = []
    prev_good_end = 0.0
    for block in merged:
        bs, be = block["start"], block["end"]
        gap = bs - prev_good_end
        # Only binary-search edges when corrupt zone is preceded by clean audio.
        if gap > COARSE_STEP_S * 2:
            bad_start = _refine_boundary(video, prev_good_end, bs, want_bad=True)
            # Search from inside the corrupt block toward clean audio after it.
            bad_end = _refine_boundary(
                video,
                max(prev_good_end, be - COARSE_STEP_S),
                min(duration, be + COARSE_STEP_S * 8),
                want_bad=False,
            )
        else:
            bad_start, bad_end = bs, min(duration, be)
        if bad_end > bad_start + 0.5:
            refined.append({"start": bad_start, "end": bad_end})
        prev_good_end = bad_end

    return refined


def load_or_scan_corrupt_ranges(
    video: Path,
    edit_dir: Path,
    duration: float,
    *,
    force_rescan: bool = False,
) -> list[dict]:
    path = cache_path(edit_dir)
    edit_dir.mkdir(parents=True, exist_ok=True)
    video = video.resolve()
    mtime = video.stat().st_mtime if video.exists() else 0

    if not force_rescan and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (
                cached.get("video") == str(video)
                and abs(float(cached.get("mtime", 0)) - mtime) < 1
                and abs(float(cached.get("duration", 0)) - duration) < 1
            ):
                return cached.get("ranges") or []
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    print(f"  scanning audio decode health ({duration / 60:.1f} min source)...", flush=True)
    ranges = scan_corrupt_ranges(video, duration)
    path.write_text(
        json.dumps(
            {
                "video": str(video),
                "mtime": mtime,
                "duration": duration,
                "ranges": ranges,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if ranges:
        total = sum(r["end"] - r["start"] for r in ranges)
        print(
            f"  WARN: {len(ranges)} corrupt audio region(s), "
            f"{total:.0f}s total — excluded from all clips",
            flush=True,
        )
        for r in ranges:
            print(f"    [{r['start']:.1f}s – {r['end']:.1f}s]", flush=True)
    else:
        print("  audio decode OK (no corrupt regions)", flush=True)
    return ranges


def range_overlaps_corrupt(start: float, end: float, corrupt: list[dict]) -> bool:
    for c in corrupt:
        if end > c["start"] and start < c["end"]:
            return True
    return False


def subtract_corrupt(
    start: float, end: float, corrupt: list[dict]
) -> list[tuple[float, float]]:
    """Return clean sub-spans of [start, end] with corrupt regions removed."""
    parts = [(start, end)]
    for c in corrupt:
        cs, ce = c["start"], c["end"]
        next_parts: list[tuple[float, float]] = []
        for ps, pe in parts:
            if pe <= cs or ps >= ce:
                next_parts.append((ps, pe))
            else:
                if ps < cs:
                    next_parts.append((ps, cs))
                if pe > ce:
                    next_parts.append((ce, pe))
        parts = next_parts
    return [(s, e) for s, e in parts if e - s > 0.08]


def filter_clean_ranges(
    ranges: list[dict],
    corrupt: list[dict],
    *,
    source_key: str,
) -> tuple[list[dict], int]:
    """Drop KEEP ranges that touch corrupt audio; return (kept, dropped_count)."""
    if not corrupt:
        return ranges, 0

    kept: list[dict] = []
    dropped = 0
    for r in ranges:
        if range_overlaps_corrupt(r["start"], r["end"], corrupt):
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def split_continuous_range(
    start: float,
    end: float,
    corrupt: list[dict],
    source_key: str,
    quote: str,
    reason: str,
) -> list[dict]:
    """One continuous span → multiple ranges, omitting corrupt gaps."""
    if not corrupt:
        return [
            {
                "source": source_key,
                "start": round(start, 3),
                "end": round(end, 3),
                "beat": "KEEP",
                "quote": quote,
                "reason": reason,
            }
        ]
    parts = subtract_corrupt(start, end, corrupt)
    return [
        {
            "source": source_key,
            "start": round(s, 3),
            "end": round(e, 3),
            "beat": "KEEP",
            "quote": quote,
            "reason": reason + " (corrupt gaps removed)",
        }
        for s, e in parts
    ]
