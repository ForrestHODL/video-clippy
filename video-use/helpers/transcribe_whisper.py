"""Transcribe video with local Whisper — Scribe-compatible JSON output.

Default transcription backend for video-use (via transcribe.py / transcribe_batch.py).
Prefers faster-whisper; falls back to whisper.cpp CLI if installed.
Output matches ElevenLabs Scribe shape so pack_transcripts.py and build_clips.py work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL = "small.en"
HF_REPO = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


def find_whisper_cli() -> Path | None:
    for name in ("whisper-cli", "whisper"):
        p = shutil.which(name)
        if p:
            return Path(p)
    for p in (
        Path(r"C:\Program Files\whisper.cpp\whisper-cli.exe"),
        Path.home() / ".cache" / "hyperframes" / "whisper" / "whisper.cpp" / "build" / "bin" / "whisper-cli.exe",
        Path.home() / ".cache" / "hyperframes" / "whisper" / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe",
    ):
        if p.exists():
            return p
    return None


def model_path(model: str) -> Path:
    cache = Path.home() / ".cache" / "hyperframes" / "whisper" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"ggml-{model}.bin"
    if dest.exists():
        return dest
    url = f"{HF_REPO}/ggml-{model}.bin"
    print(f"  downloading model {model}...")
    subprocess.run(
        ["curl", "-L", "-o", str(dest), url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dest


def extract_wav(video: Path, wav: Path, start: float = 0.0, duration: float | None = None) -> bool:
    """Extract PCM; returns True if usable audio was written (partial OK on corrupt AAC)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+discardcorrupt+genpts",
        "-err_detect",
        "ignore_err",
        "-i",
        str(video),
    ]
    if start > 0:
        cmd.extend(["-ss", str(start)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    subprocess.run(cmd, capture_output=True)
    return wav.exists() and wav.stat().st_size > 5000


def run_whisper(wav: Path, out_base: Path, model: str, language: str | None) -> Path:
    cli = find_whisper_cli()
    if not cli:
        sys.exit(
            "whisper-cli not found. Install whisper.cpp or run: brew install whisper-cpp\n"
            "HyperFrames will auto-build on first `npx hyperframes transcribe`."
        )
    mp = model_path(model)
    args = [
        str(cli),
        "--model", str(mp),
        "--output-json-full",
        "--output-file", str(out_base),
        "--dtw", model,
        "--suppress-nst",
    ]
    if language:
        args.extend(["--language", language])
    args.append(str(wav))
    print(f"  whisper transcribing ({model})...")
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    out_json = Path(f"{out_base}.json")
    if not out_json.exists():
        sys.exit(f"whisper produced no output at {out_json}")
    return out_json


def parse_whisper_cpp(data: dict) -> list[dict]:
    """Parse whisper.cpp --output-json-full into [{text, start, end}, ...] seconds."""
    words: list[dict] = []
    for seg in data.get("transcription") or []:
        for tok in seg.get("tokens") or []:
            text = (tok.get("text") or "").strip()
            if not text or text.startswith("[_") or text.startswith("[BLANK"):
                continue
            off = tok.get("offsets") or {}
            start = float(off.get("from", 0)) / 1000.0
            end = float(off.get("to", start)) / 1000.0
            if end <= start:
                end = start + 0.05
            words.append({"text": text, "start": start, "end": end})
    return words


def merge_whisper_fragments(words: list[dict]) -> list[dict]:
    """Rejoin single-letter fragments (whisper tokenization quirk)."""
    i = 0
    while i < len(words) - 1:
        cur, nxt = words[i], words[i + 1]
        ct, nt = cur["text"], nxt["text"]
        merge = (
            len(ct) == 1 and ct.isupper() and ct not in "IAO" and nt and nt[0].islower()
        ) or (ct.endswith("in") and nt.lower() == "in'")
        if merge:
            cur["text"] = ct + nt
            cur["end"] = nxt["end"]
            words.pop(i + 1)
        else:
            i += 1
    return words


def interpolate_zero_duration(words: list[dict]) -> None:
    for i, w in enumerate(words):
        if w["end"] > w["start"]:
            continue
        j = i
        while j < len(words) and words[j]["end"] <= words[j]["start"]:
            j += 1
        prev_end = words[i - 1]["end"] if i > 0 else w["start"]
        next_start = words[j]["start"] if j < len(words) else prev_end + (j - i) * 0.3
        span = max(0.1, next_start - prev_end)
        per = span / max(1, j - i)
        for k in range(i, j):
            words[k]["start"] = round(prev_end + (k - i) * per, 3)
            words[k]["end"] = round(prev_end + (k - i + 1) * per, 3)


def to_scribe_words(words: list[dict]) -> list[dict]:
    """Convert plain words to Scribe token list with spacing entries."""
    out: list[dict] = []
    for i, w in enumerate(words):
        out.append(
            {
                "text": w["text"],
                "start": w["start"],
                "end": w["end"],
                "type": "word",
                "speaker_id": "speaker_0",
            }
        )
        if i + 1 < len(words):
            gap_s = w["end"]
            gap_e = words[i + 1]["start"]
            if gap_e > gap_s + 0.001:
                out.append(
                    {
                        "text": " ",
                        "start": gap_s,
                        "end": gap_e,
                        "type": "spacing",
                        "speaker_id": "speaker_0",
                    }
                )
    return out


def build_scribe_json(words: list[dict]) -> dict:
    merge_whisper_fragments(words)
    interpolate_zero_duration(words)
    scribe_words = to_scribe_words(words)
    text = " ".join(w["text"] for w in words)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
    return {
        "language_code": "eng",
        "language_probability": 1.0,
        "text": text,
        "words": scribe_words,
        "source": "whisper.cpp",
    }


def probe_duration(video_path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout.strip())


def run_faster_whisper_on_path(
    audio_path: Path,
    model: str,
    language: str | None,
    label: str = "",
) -> list[dict]:
    from faster_whisper import WhisperModel

    size = model.replace(".en", "") if model.endswith(".en") else model
    tag = f" ({label})" if label else ""
    print(f"  faster-whisper{tag} ({size})...", flush=True)
    wm = WhisperModel(size, device="cpu", compute_type="int8")
    lang = language or "en"
    segments, _info = wm.transcribe(
        str(audio_path),
        language=lang,
        word_timestamps=True,
        vad_filter=False,
    )
    words: list[dict] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append({"text": text, "start": w.start, "end": w.end})
        elif seg.text.strip():
            words.append({"text": seg.text.strip(), "start": seg.start, "end": seg.end})
    return words


def run_faster_whisper(video_path: Path, model: str, language: str | None) -> list[dict]:
    return run_faster_whisper_on_path(video_path, model, language)


def clean_spans(duration: float, corrupt: list[dict]) -> list[tuple[float, float]]:
    """Return (start, duration) spans with decodable audio."""
    spans: list[tuple[float, float]] = []
    pos = 0.0
    for block in sorted(corrupt, key=lambda c: c["start"]):
        if block["start"] > pos + 2.0:
            spans.append((pos, block["start"] - pos))
        pos = max(pos, block["end"])
    if pos < duration - 2.0:
        spans.append((pos, duration - pos))
    return [(s, d) for s, d in spans if d >= 4.0]


def merge_word_lists(*lists: list[dict]) -> list[dict]:
    seen: set[tuple[str, float]] = set()
    out: list[dict] = []
    for words in lists:
        for w in words:
            key = (w["text"], round(w["start"], 2))
            if key in seen:
                continue
            seen.add(key)
            out.append(w)
    out.sort(key=lambda w: w["start"])
    return out


def transcribe_time_windows(
    video_path: Path,
    duration: float,
    model: str,
    language: str | None,
    *,
    window_s: float = 420.0,
    overlap_s: float = 15.0,
) -> list[dict]:
    """Transcribe overlapping windows — no corrupt-audio scan."""
    words: list[dict] = []
    start = 0.0
    i = 0
    while start < duration - 2.0:
        i += 1
        chunk_dur = min(window_s, duration - start)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / f"win_{i:02d}.wav"
            if not extract_wav(video_path, wav, start, chunk_dur):
                print(f"    window {i} skip (no audio at {start:.0f}s)", flush=True)
                start += window_s - overlap_s
                continue
            chunk_words = run_faster_whisper_on_path(
                wav, model, language, label=f"window {i} @ {start:.0f}s"
            )
            for w in chunk_words:
                words.append(
                    {
                        "text": w["text"],
                        "start": w["start"] + start,
                        "end": w["end"] + start,
                    }
                )
            print(f"    window {i}: +{len(chunk_words)} words", flush=True)
        if start + chunk_dur >= duration - 1:
            break
        start += window_s - overlap_s
    return merge_word_lists(words)


def transcribe_clean_chunks(
    video_path: Path,
    duration: float,
    corrupt: list[dict],
    model: str,
    language: str | None,
) -> list[dict]:
    from audio_corruption import probe_extract_ok

    words: list[dict] = []
    spans = clean_spans(duration, corrupt)
    if not spans:
        # Fall back: probe every 60s windows that decode
        t = 0.0
        while t < duration - 4:
            if probe_extract_ok(video_path, t, 8.0):
                spans.append((t, min(120.0, duration - t)))
            t += 60.0
    for i, (start, dur) in enumerate(spans, 1):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / f"chunk_{i:02d}.wav"
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ss",
                    str(start),
                    "-t",
                    str(dur),
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
            if r.returncode != 0 or not wav.exists():
                print(f"    chunk {i} skip (extract failed at {start:.0f}s)", flush=True)
                continue
            chunk_words = run_faster_whisper_on_path(
                wav, model, language, label=f"chunk {i} @ {start:.0f}s"
            )
            for w in chunk_words:
                words.append(
                    {
                        "text": w["text"],
                        "start": w["start"] + start,
                        "end": w["end"] + start,
                    }
                )
            print(f"    chunk {i}: +{len(chunk_words)} words", flush=True)
    words.sort(key=lambda w: w["start"])
    return words


def transcribe_video(
    video_path: Path,
    edit_dir: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = "en",
    *,
    force: bool = False,
) -> Path:
    edit_dir.mkdir(parents=True, exist_ok=True)
    out_path = edit_dir / "transcripts" / f"{video_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"  cached: {out_path.name}")
        return out_path

    print(f"  transcribing {video_path.name}")
    duration = probe_duration(video_path)
    words: list[dict] = []
    if find_whisper_cli():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            wav = tmp_dir / "audio.wav"
            if not extract_wav(video_path, wav):
                sys.exit(f"could not extract audio from {video_path.name}")
            raw = run_whisper(wav, tmp_dir / "transcript", model, language)
            data = json.loads(raw.read_text(encoding="utf-8"))
            words = parse_whisper_cpp(data)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            if extract_wav(video_path, wav):
                words = run_faster_whisper_on_path(wav, model, language)
            else:
                words = run_faster_whisper(video_path, model, language)

    last_end = words[-1]["end"] if words else 0.0
    need_more = last_end < duration * 0.82 and not find_whisper_cli()
    if need_more:
        print(
            f"  partial transcript ({last_end:.0f}s / {duration:.0f}s) — windowed re-transcribe",
            flush=True,
        )
        extra = transcribe_time_windows(video_path, duration, model, language)
        words = merge_word_lists(words, extra)

    if not words:
        sys.exit("whisper returned no words")
    scribe = build_scribe_json(words)
    src = "faster-whisper-windowed" if need_more else (
        "faster-whisper" if not find_whisper_cli() else "whisper.cpp"
    )
    scribe["source"] = src
    out_path.write_text(json.dumps(scribe, indent=2), encoding="utf-8")
    print(f"  saved: {out_path.name} ({len(words)} words, span 0–{words[-1]['end']:.1f}s)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Whisper transcription → Scribe-compatible JSON")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()
    transcribe_video(args.video.resolve(), args.edit_dir.resolve(), args.model, args.language)


if __name__ == "__main__":
    main()
