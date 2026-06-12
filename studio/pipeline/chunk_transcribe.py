"""Chunk transcribe with output-side seek (works on tricky AAC sources)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from studio.paths import helpers_dir

sys.path.insert(0, str(helpers_dir()))
from transcribe_whisper import (  # noqa: E402
    build_scribe_json,
    merge_word_lists,
    probe_duration,
    run_faster_whisper_on_path,
    transcribe_time_windows,
)


def extract(video: Path, start: float, dur: float, wav: Path) -> bool:
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-i", str(video),
            "-t", str(dur), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(wav),
        ],
        capture_output=True,
    )
    return r.returncode == 0 and wav.exists() and wav.stat().st_size > 5000


def chunk_transcribe(video: Path, edit_dir: Path, *, chunk_s: float = 280.0) -> Path:
    edit_dir.mkdir(parents=True, exist_ok=True)
    out = edit_dir / "transcripts" / f"{video.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video)
    words: list[dict] = []

    intro = transcribe_time_windows(video, min(duration, 420), "small.en", "en", window_s=300, overlap_s=20)
    words = merge_word_lists(intro)
    last = words[-1]["end"] if words else 0.0

    step = chunk_s - 20
    start = max(0.0, last - 15) if last > 30 else 0.0
    i = 0
    while start < duration - 5:
        i += 1
        dur = min(chunk_s, duration - start)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / f"c{i:02d}.wav"
            if not extract(video, start, dur, wav):
                print(f"  skip chunk @ {start:.0f}s", flush=True)
                start += step
                continue
            cw = run_faster_whisper_on_path(wav, "small.en", "en", label=f"{start:.0f}s")
            for w in cw:
                words.append({"text": w["text"], "start": w["start"] + start, "end": w["end"] + start})
            print(f"  chunk {i} @ {start:.0f}s: +{len(cw)} words", flush=True)
        if start + dur >= duration - 3:
            break
        start += step

    words = merge_word_lists(words)
    scribe = build_scribe_json(words)
    scribe["source"] = "faster-whisper-chunked"
    out.write_text(json.dumps(scribe, indent=2), encoding="utf-8")
    alias = re.sub(r"[^a-z0-9]+", "_", video.stem.lower()).strip("_")
    alias_path = edit_dir / "transcripts" / f"{alias}.json"
    if alias_path != out:
        alias_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    end_s = words[-1]["end"] if words else 0
    print(f"  saved {out.name} ({len(words)} words, 0–{end_s:.0f}s)")
    return out
