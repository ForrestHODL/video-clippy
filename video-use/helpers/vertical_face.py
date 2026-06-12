"""Reframe landscape video to 9:16 vertical with smoothed face tracking.

Detects the largest face per sampled frame, smooths the crop center, and
pans a 9:16 window to keep the speaker centered.

Modes:
  gentle — default via --vertical; slow subtle drift toward face
  full   — responsive tracking (--vertical-face)

Usage:
    python helpers/vertical_face.py <input.mp4> -o <output.mp4>
    python helpers/vertical_face.py <input.mp4> -o <out.mp4> --mode gentle
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from gpu_encode import video_encode_args

OUT_W, OUT_H = 1080, 1920
TrackMode = Literal["full", "gentle"]


@dataclass(frozen=True)
class TrackSettings:
    sample_every: int
    smooth: float
    max_step: float
    max_step_y_ratio: float
    default_x_ratio: float
    default_y_ratio: float


SETTINGS: dict[TrackMode, TrackSettings] = {
    "full": TrackSettings(
        sample_every=2,
        smooth=0.12,
        max_step=18.0,
        max_step_y_ratio=0.5,
        default_x_ratio=0.5,
        default_y_ratio=0.38,
    ),
    "gentle": TrackSettings(
        sample_every=4,
        smooth=0.035,
        max_step=4.0,
        max_step_y_ratio=0.25,
        default_x_ratio=0.40,
        default_y_ratio=0.38,
    ),
}


def _load_detector() -> cv2.CascadeClassifier:
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(str(path))
    if det.empty():
        sys.exit(f"failed to load face cascade: {path}")
    return det


def _detect_face_center(gray: np.ndarray, detector: cv2.CascadeClassifier) -> tuple[float, float] | None:
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x + w / 2.0, y + h / 2.0


def _crop_size(src_w: int, src_h: int) -> tuple[int, int]:
    """9:16 crop that fits inside the source frame."""
    crop_h = src_h
    crop_w = int(round(crop_h * 9 / 16))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w * 16 / 9))
    return crop_w, crop_h


def _default_center(
    src_w: int,
    src_h: int,
    key_x: list[float],
    key_y: list[float],
    settings: TrackSettings,
) -> tuple[float, float]:
    if key_x:
        return float(np.median(key_x)), float(np.median(key_y))
    return src_w * settings.default_x_ratio, src_h * settings.default_y_ratio


def _interpolate_centers(
    key_idx: list[int],
    key_x: list[float],
    key_y: list[float],
    total: int,
    default_x: float,
    default_y: float,
) -> tuple[list[float], list[float]]:
    if not key_idx:
        return [default_x] * total, [default_y] * total

    xs = [default_x] * total
    ys = [default_y] * total
    for i, xi, yi in zip(key_idx, key_x, key_y):
        xs[i] = xi
        ys[i] = yi

    for i in range(total):
        if i in key_idx:
            continue
        prev_keys = [k for k in key_idx if k <= i]
        next_keys = [k for k in key_idx if k >= i]
        if not prev_keys and not next_keys:
            continue
        if not prev_keys:
            xs[i], ys[i] = xs[next_keys[0]], ys[next_keys[0]]
            continue
        if not next_keys:
            xs[i], ys[i] = xs[prev_keys[-1]], ys[prev_keys[-1]]
            continue
        p, n = prev_keys[-1], next_keys[0]
        if p == n:
            xs[i], ys[i] = xs[p], ys[p]
        else:
            t = (i - p) / (n - p)
            xs[i] = xs[p] * (1 - t) + xs[n] * t
            ys[i] = ys[p] * (1 - t) + ys[n] * t
    return xs, ys


def _smooth_trajectory(
    xs: list[float],
    ys: list[float],
    settings: TrackSettings,
) -> tuple[list[float], list[float]]:
    sx, sy = xs[0], ys[0]
    out_x, out_y = [sx], [sy]
    max_dy = settings.max_step * settings.max_step_y_ratio
    for tx, ty in zip(xs[1:], ys[1:]):
        dx = max(-settings.max_step, min(settings.max_step, tx - sx))
        dy = max(-max_dy, min(max_dy, ty - sy))
        sx += settings.smooth * dx
        sy += settings.smooth * dy
        out_x.append(sx)
        out_y.append(sy)
    return out_x, out_y


def analyze_trajectory(
    cap: cv2.VideoCapture,
    detector: cv2.CascadeClassifier,
    frame_count: int,
    mode: TrackMode = "full",
) -> tuple[list[float], list[float], int, int]:
    settings = SETTINGS[mode]
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    key_idx: list[int] = []
    key_x: list[float] = []
    key_y: list[float] = []

    for i in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        if i % settings.sample_every != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        center = _detect_face_center(gray, detector)
        if center:
            key_idx.append(i)
            key_x.append(center[0])
            key_y.append(center[1])

    default_x, default_y = _default_center(src_w, src_h, key_x, key_y, settings)
    raw_x, raw_y = _interpolate_centers(key_idx, key_x, key_y, frame_count, default_x, default_y)
    return _smooth_trajectory(raw_x, raw_y, settings)


def _crop_frame(
    frame: np.ndarray,
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    x = int(round(max(0, min(cx - crop_w / 2, src_w - crop_w))))
    y = int(round(max(0, min(cy - crop_h / 2, src_h - crop_h))))
    cropped = frame[y : y + crop_h, x : x + crop_w]
    return cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)


def reframe_video(
    source: Path,
    out_path: Path,
    start: float = 0.0,
    duration: float | None = None,
    profile: str = "final",
    force_cpu: bool = False,
    mode: TrackMode = "full",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        segment = tmp_dir / "segment.mp4"
        silent = tmp_dir / "silent.mp4"

        trim = ["-ss", f"{start:.3f}"]
        if duration is not None:
            trim += ["-t", f"{duration:.3f}"]

        subprocess.run(
            [
                "ffmpeg", "-y",
                *trim,
                "-i", str(source),
                "-an",
                *video_encode_args("high", force_cpu),
                str(segment),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        cap = cv2.VideoCapture(str(segment))
        if not cap.isOpened():
            sys.exit(f"cannot open segment: {segment}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop_w, crop_h = _crop_size(src_w, src_h)

        detector = _load_detector()
        label = "gentle face track" if mode == "gentle" else "face track"
        print(
            f"  {label}: {frame_count} frames @ {fps:.1f}fps, "
            f"crop {crop_w}x{crop_h} → {OUT_W}x{OUT_H}"
        )
        xs, ys = analyze_trajectory(cap, detector, frame_count, mode=mode)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(silent), fourcc, fps, (OUT_W, OUT_H))
        if not writer.isOpened():
            sys.exit("VideoWriter failed — check ffmpeg/mp4v support")

        for i in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(_crop_frame(frame, xs[i], ys[i], crop_w, crop_h))
        writer.release()
        cap.release()

        fade_d = 0.03
        dur = frame_count / fps if fps else 0
        fade_out = max(0.0, dur - fade_d)
        af = f"afade=t=in:st=0:d={fade_d},afade=t=out:st={fade_out:.3f}:d={fade_d}"

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(silent),
                *trim,
                "-i", str(source),
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-af", af,
                *video_encode_args(profile, force_cpu),
                "-r", f"{fps:.3f}",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-shortest",
                "-movflags", "+faststart",
                str(out_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Vertical 9:16 reframe with face tracking")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument(
        "--mode",
        choices=("full", "gentle"),
        default="full",
        help="full = responsive pan; gentle = slow subtle drift toward face",
    )
    args = ap.parse_args()
    reframe_video(
        args.input,
        args.output,
        start=args.start,
        duration=args.duration,
        mode=args.mode,  # type: ignore[arg-type]
    )
    mb = args.output.stat().st_size / (1024 * 1024)
    print(f"done: {args.output} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
