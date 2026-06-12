"""Batch-transcribe every video in a directory.

Default: local Whisper (faster-whisper). Use --elevenlabs for Scribe API.

Walks <videos_dir> for common video extensions, writes transcripts to
<videos_dir>/edit/transcripts/<name>.json. Cached per-file.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --model small.en
    python helpers/transcribe_batch.py <videos_dir> --elevenlabs --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import load_api_key, transcribe_one


VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    return videos


def transcribe_whisper_one(
    video: Path,
    edit_dir: Path,
    model: str,
    language: str | None,
) -> Path:
    from transcribe_whisper import transcribe_video

    return transcribe_video(video, edit_dir, model=model, language=language or "en")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel batch transcription of a videos directory")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: 1 for Whisper, 4 for ElevenLabs)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="small.en",
        help="Whisper model when not using --elevenlabs (default: small.en)",
    )
    ap.add_argument(
        "--elevenlabs",
        action="store_true",
        help="Use ElevenLabs Scribe instead of local Whisper",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="ElevenLabs only: optional speaker count for diarization.",
    )
    args = ap.parse_args()

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")

    already_cached = [v for v in videos if (edit_dir / "transcripts" / f"{v.stem}.json").exists()]
    pending = [v for v in videos if v not in already_cached]

    provider = "elevenlabs" if args.elevenlabs else "whisper"
    workers = args.workers if args.workers is not None else (4 if args.elevenlabs else 1)

    print(f"found {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    api_key: str | None = None
    if args.elevenlabs:
        api_key = load_api_key()

    print(f"transcribing {len(pending)} files via {provider} ({workers} worker(s))")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if args.elevenlabs:
            futures = {
                pool.submit(
                    transcribe_one,
                    video=v,
                    edit_dir=edit_dir,
                    api_key=api_key,  # type: ignore[arg-type]
                    language=args.language,
                    num_speakers=args.num_speakers,
                    verbose=False,
                ): v
                for v in pending
            }
        else:
            futures = {
                pool.submit(
                    transcribe_whisper_one,
                    video=v,
                    edit_dir=edit_dir,
                    model=args.model,
                    language=args.language,
                ): v
                for v in pending
            }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  →  {out.name}")
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
