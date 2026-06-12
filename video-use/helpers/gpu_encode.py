"""GPU video encoding helpers — AMD AMF (h264_amf) on Radeon, with libx264 fallback.

Set VIDEO_USE_GPU=0 to force CPU (libx264). Default is auto: use AMF when ffmpeg
lists h264_amf and a quick probe succeeds.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

Profile = str  # draft | preview | final | high


@lru_cache(maxsize=1)
def amf_available() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if "h264_amf" not in (out.stdout or ""):
            return False
        probe = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=640x360:d=0.1",
                "-frames:v", "5", "-c:v", "h264_amf", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def prefer_gpu(force_cpu: bool = False) -> bool:
    if force_cpu:
        return False
    env = os.environ.get("VIDEO_USE_GPU", "auto").strip().lower()
    if env in ("0", "false", "no", "cpu", "off"):
        return False
    if env in ("1", "true", "yes", "gpu", "amf", "on"):
        return True
    return amf_available()


def encoder_label(force_cpu: bool = False) -> str:
    return "h264_amf (GPU)" if prefer_gpu(force_cpu) else "libx264 (CPU)"


def video_encode_args(profile: Profile = "final", force_cpu: bool = False) -> list[str]:
    """Return ffmpeg video encode arguments for the given quality profile."""
    if prefer_gpu(force_cpu):
        amf: dict[str, list[str]] = {
            "draft": [
                "-c:v", "h264_amf",
                "-usage", "transcoding", "-quality", "speed",
                "-rc", "cqp", "-qp_i", "28", "-qp_p", "30",
            ],
            "preview": [
                "-c:v", "h264_amf",
                "-usage", "transcoding", "-quality", "balanced",
                "-rc", "cqp", "-qp_i", "24", "-qp_p", "26",
            ],
            "final": [
                "-c:v", "h264_amf",
                "-usage", "high_quality", "-quality", "quality",
                "-rc", "cqp", "-qp_i", "20", "-qp_p", "22",
            ],
            "high": [
                "-c:v", "h264_amf",
                "-usage", "high_quality", "-quality", "quality",
                "-rc", "cqp", "-qp_i", "18", "-qp_p", "20",
            ],
        }
        return amf.get(profile, amf["final"]) + ["-pix_fmt", "yuv420p"]

    x264: dict[str, list[str]] = {
        "draft": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"],
        "preview": ["-c:v", "libx264", "-preset", "medium", "-crf", "22"],
        "final": ["-c:v", "libx264", "-preset", "fast", "-crf", "20"],
        "high": ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
    }
    return x264.get(profile, x264["final"]) + ["-pix_fmt", "yuv420p"]
