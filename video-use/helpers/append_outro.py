"""Append a shared outro clip after a rendered body — fade-out + loudness match.

Default behavior (required for all clip builds that use an outro):
  - Fade body video to black and audio out over the last ~1.25s (smooth handoff)
  - Brief fade-in on outro video/audio (~0.3s)
  - Match outro mean loudness to the body (outro must never sound louder than the clip)

Usage:
  python append_outro.py body.mp4 outro.mp4 -o final.mp4
  python append_outro.py body.mp4 outro.mp4 -o final.mp4 --draft
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_encode import video_encode_args  # noqa: E402

DEFAULT_FADE_S = 1.25
OUTRO_FADE_IN_S = 0.35
OUTRO_AUDIO_FADE_IN_S = 0.25


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def probe_video_size(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(r.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def measure_mean_volume_db(path: Path) -> float:
    """Return mean_volume in dB from ffmpeg volumedetect (full file)."""
    r = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (r.stderr or "") + (r.stdout or "")
    m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", text)
    if not m:
        return -20.0
    return float(m.group(1))


def outro_gain_db(body_db: float, outro_db: float) -> float:
    """dB gain for outro audio so it matches body mean level (never louder than clip)."""
    gain = body_db - outro_db
    # Never boost outro above body peak perception — cap boost at +3 dB
    return min(gain, 3.0)


def probe_has_audio(path: Path) -> bool:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(json.loads(r.stdout).get("streams"))


def append_outro(
    body_mp4: Path,
    outro_mp4: Path,
    output_mp4: Path,
    *,
    draft: bool = False,
    fade_s: float = DEFAULT_FADE_S,
    delete_body: bool = False,
) -> None:
    if not body_mp4.exists():
        raise FileNotFoundError(body_mp4)
    if not outro_mp4.exists():
        raise FileNotFoundError(outro_mp4)

    duration = probe_duration(body_mp4)
    fade_s = min(fade_s, max(0.5, duration * 0.15))
    fade_start = max(0.0, duration - fade_s)

    body_db = measure_mean_volume_db(body_mp4) if probe_has_audio(body_mp4) else -20.0
    outro_db = measure_mean_volume_db(outro_mp4)
    gain_db = outro_gain_db(body_db, outro_db)

    body_w, body_h = probe_video_size(body_mp4)
    w, h = body_w, body_h
    if draft and w > 1280:
        w, h = 1280, 720

    body_vprep = ""
    if (body_w, body_h) != (w, h):
        body_vprep = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        )

    has_body_audio = probe_has_audio(body_mp4)
    outro_v = (
        f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
        f"fade=t=in:st=0:d={OUTRO_FADE_IN_S:.3f},setsar=1[v1]"
    )
    outro_a = (
        f"[1:a]volume={gain_db:.2f}dB,"
        f"afade=t=in:st=0:d={OUTRO_AUDIO_FADE_IN_S:.3f}[a1]"
    )
    if has_body_audio:
        filter_parts = [
            f"[0:v]{body_vprep}fade=t=out:st={fade_start:.3f}:d={fade_s:.3f}:color=black[v0]",
            f"[0:a]afade=t=out:st={fade_start:.3f}:d={fade_s:.3f}[a0]",
            outro_v,
            outro_a,
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
        ]
    else:
        filter_parts = [
            f"[0:v]{body_vprep}fade=t=out:st={fade_start:.3f}:d={fade_s:.3f}:color=black[v0]",
            outro_v,
            "[v0][v1]concat=n=2:v=1:a=0[v]",
            outro_a,
        ]
    fc = ";".join(filter_parts)

    profile = "draft" if draft else "final"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(body_mp4),
        "-i",
        str(outro_mp4),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
    ]
    if has_body_audio:
        cmd += ["-map", "[a]"]
    else:
        cmd += ["-map", "[a1]"]
    cmd += [
        *video_encode_args(profile),
        "-r",
        "24",
        "-c:a",
        "aac",
        "-b:a",
        "192k" if not draft else "128k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        enc_start = cmd.index("-filter_complex")
        enc_end = cmd.index("-map")
        cmd_cpu = (
            cmd[:enc_end]
            + ["-map", "[v]"]
            + (["-map", "[a]"] if has_body_audio else ["-map", "[a1]"])
            + list(video_encode_args(profile, force_cpu=True))
            + ["-r", "24", "-c:a", "aac", "-b:a", "192k" if not draft else "128k", "-ar", "48000"]
            + ["-movflags", "+faststart", str(output_mp4)]
        )
        subprocess.run(cmd_cpu, check=True, capture_output=True)

    if delete_body:
        body_mp4.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Append outro with fade-out and loudness match")
    ap.add_argument("body", type=Path, help="Rendered clip body (no outro)")
    ap.add_argument("outro", type=Path, help="Shared outro mp4")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--draft", action="store_true", help="720p-style fast encode")
    ap.add_argument("--fade", type=float, default=DEFAULT_FADE_S, help="Body fade-out seconds")
    ap.add_argument("--delete-body", action="store_true")
    args = ap.parse_args()
    append_outro(
        args.body,
        args.outro,
        args.output,
        draft=args.draft,
        fade_s=args.fade,
        delete_body=args.delete_body,
    )
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
