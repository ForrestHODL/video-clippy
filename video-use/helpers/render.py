"""Render a video from an EDL.

Implements the HEURISTICS render pipeline in the correct order:

  1. Per-segment extract with color grade + 30ms audio fades baked in
  2. Lossless -c copy concat into base.mp4
  3. If overlays or subtitles: single filter graph that overlays animations
     (with PTS shift so frame 0 lands at the overlay window start)
     and applies `subtitles` filter LAST → final.mp4

Optionally builds a master SRT from the per-source transcripts + EDL
output-timeline offsets, applies the proven force_style (2-word
UPPERCASE chunks, Helvetica 18 Bold, MarginV=35).

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
    from gpu_encode import encoder_label, prefer_gpu, video_encode_args
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}

    def prefer_gpu(force_cpu: bool = False) -> bool:
        return False

    def encoder_label(force_cpu: bool = False) -> str:
        return "libx264 (CPU)"

    def video_encode_args(profile: str = "final", force_cpu: bool = False) -> list[str]:
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]


# -------- Subtitle style (bold-overlay, proven at 1920×1080 and 1080×1920) --
#
# MarginV is NOT taste — it is a platform safe-zone rule.
# TikTok / IG Reels / Shorts UI (caption, username, music, right-rail actions)
# covers roughly the bottom ~25–30% of a 1080×1920 frame. Captions placed near
# the bottom edge get clipped or obscured by the UI. libass auto-scales the
# render canvas relative to PlayResY=288, so MarginV=90 lands the caption
# baseline roughly 30% up from the bottom on any aspect — clear of the UI on
# every major vertical-video platform. Do not drop this below ~75 without a
# specific reason.
SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=90"
)

# 9:16 vertical — lower-third, large mobile-readable captions
# FontSize is in PlayRes pixels (1920-tall); 100 ≈ 5% of frame height
VERTICAL_SUB_FORCE_STYLE = (
    "PlayResX=1080,PlayResY=1920,"
    "FontName=Arial,FontSize=100,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
    "BorderStyle=1,Outline=6,Shadow=2,"
    "Alignment=2,MarginV=300"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True)


def resolve_grade_filter(grade_field: str | None) -> str:
    """The EDL's 'grade' field can be a preset name, a raw ffmpeg filter, or 'auto'.

    Returns the filter string to embed into the per-segment -vf chain.
    For 'auto', returns the sentinel "__AUTO__" which is resolved per-segment.
    """
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    # Preset names are short identifiers, filter strings contain '=' or ','.
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


def resolve_project_dir(edl_dir: Path) -> Path:
    """EDL may live in a subfolder (e.g. edls/) — project root holds transcripts."""
    if (edl_dir / "transcripts").is_dir():
        return edl_dir
    parent = edl_dir.parent
    if (parent / "transcripts").is_dir():
        return parent
    return edl_dir


def resolve_transcripts_dir(project_dir: Path) -> Path:
    return project_dir / "transcripts"


def escape_ffmpeg_subtitles_path(p: Path) -> str:
    """Escape a path for ffmpeg's subtitles filter on Windows (drive colons)."""
    s = str(p.resolve()).replace("\\", "/")
    if re.match(r"^[A-Za-z]:", s):
        s = s[0] + r"\:" + s[2:]
    return s.replace("'", r"\'")


def stage_srt_for_ffmpeg(srt_path: Path) -> tuple[Path, str]:
    """Copy SRT to temp dir; return (cwd, relative filename) for ffmpeg subtitles filter."""
    staged = Path(tempfile.gettempdir()) / "vu_subs.srt"
    shutil.copy2(srt_path, staged)
    return staged.parent, staged.name


def resolve_path(maybe_path: str, base: Path) -> Path:
    """Resolve a path that may be absolute or relative to `base`."""
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


STREAM_COPY_MIN_DURATION = 90.0

FFMPEG_CORRUPT_INPUT = ["-fflags", "+discardcorrupt+genpts", "-err_detect", "ignore_err"]


def probe_has_audio(path: Path) -> bool:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return bool(json.loads(r.stdout).get("streams"))


def probe_video_size(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(r.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def should_stream_copy_first(duration: float, range_meta: dict) -> bool:
    if range_meta.get("stream_copy"):
        return True
    reason = (range_meta.get("reason") or "").lower()
    return duration >= STREAM_COPY_MIN_DURATION and (
        "timed segment" in reason or "continuous take" in reason
    )


def probe_output_seek_audio_ok(source: Path, start_s: float, duration_s: float = 2.0) -> bool:
    """True when AAC decodes with fast output seek (-ss before -i)."""
    r = subprocess.run(
        [
            "ffmpeg", "-hide_banner", *FFMPEG_CORRUPT_INPUT,
            "-ss", f"{start_s:.3f}", "-i", str(source),
            "-t", f"{duration_s:.3f}", "-vn", "-f", "null", "-",
        ],
        capture_output=True,
    )
    return r.returncode == 0


def find_first_decodable_audio(source: Path, seg_start: float, seg_end: float) -> float | None:
    """First source timestamp in [seg_start, seg_end) where output-seek audio decodes."""
    if probe_output_seek_audio_ok(source, seg_start):
        return seg_start
    if not probe_output_seek_audio_ok(source, max(seg_start, seg_end - 3.0)):
        return None
    lo, hi = seg_start, seg_end
    while hi - lo > 2.0:
        mid = (lo + hi) / 2
        if probe_output_seek_audio_ok(source, mid):
            hi = mid
        else:
            lo = mid
    return round(hi, 3)


def probe_video_stream_start(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=start_time",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def segment_has_decodable_video(path: Path, min_bytes: int = 50_000) -> bool:
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "video"


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stream_copy_segment(
    source: Path,
    seg_start: float,
    duration: float,
    out_path: Path,
    *,
    with_audio: bool,
) -> bool:
    cmd = [
        "ffmpeg", "-y", *FFMPEG_CORRUPT_INPUT,
        "-ss", f"{seg_start:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-c:v", "copy",
    ]
    if with_audio:
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out_path)]
    try:
        _run_ffmpeg(cmd)
    except subprocess.CalledProcessError:
        out_path.unlink(missing_ok=True)
        return False
    if not segment_has_decodable_video(out_path):
        out_path.unlink(missing_ok=True)
        return False
    if with_audio and not probe_has_audio(out_path):
        out_path.unlink(missing_ok=True)
        return False
    return True


def mux_delayed_audio(
    video_path: Path,
    audio_path: Path,
    delay_s: float,
    out_path: Path,
) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-itsoffset", f"{max(0.0, delay_s):.3f}",
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy", "-shortest",
        "-movflags", "+faststart", str(out_path),
    ])


def extract_audio_tail(
    source: Path, audio_start: float, duration: float, out_path: Path,
) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y", *FFMPEG_CORRUPT_INPUT,
        "-ss", f"{audio_start:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}", "-vn",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(out_path),
    ])


def try_stream_copy_extract(
    source: Path,
    seg_start: float,
    duration: float,
    out_path: Path,
) -> bool:
    """4K stream copy with AAC; recover audio after corrupt prefix via delayed mux."""
    if stream_copy_segment(source, seg_start, duration, out_path, with_audio=True):
        return True

    vonly = out_path.with_suffix(".vonly.mp4")
    if not stream_copy_segment(source, seg_start, duration, vonly, with_audio=False):
        return False

    audio_at = find_first_decodable_audio(source, seg_start, seg_start + duration)
    if audio_at is None:
        vonly.replace(out_path)
        return True

    v_start = probe_video_stream_start(vonly)
    timeline = v_start if v_start > seg_start + 0.5 else seg_start
    delay = max(0.0, audio_at - timeline)
    vid_dur = probe_duration(vonly)
    atail = out_path.with_suffix(".atail.m4a")
    try:
        extract_audio_tail(source, audio_at, vid_dur, atail)
        mux_delayed_audio(vonly, atail, delay, out_path)
    finally:
        vonly.unlink(missing_ok=True)
        atail.unlink(missing_ok=True)
    return segment_has_decodable_video(out_path) and probe_has_audio(out_path)


# -------- HDR → SDR tone mapping (HLG / PQ sources) --------------------------
#
# iPhone defaults to HLG HDR in Rec.2020 (and many mirrorless cameras ship PQ).
# If the source is HDR and we only downconvert bit depth (yuv420p10le → yuv420p)
# without tone-mapping, the output is 8-bit but still carries HLG/PQ transfer
# metadata. Players that honor the metadata (screen recorders, most social
# upload re-encodes) interpret 8-bit values in an HDR container and the result
# looks oversaturated / blown out. QuickTime on macOS can hide this locally —
# screen recording and uploaded renders cannot.
#
# Fix: detect HDR via color_transfer and prepend a zscale+tonemap chain to the
# vf graph so the output is clean Rec.709 SDR.

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG

TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,"
    "format=gbrpf32le,"
    "zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,"
    "format=yuv420p"
)


def is_hdr_source(video: Path) -> bool:
    """Return True if the source uses a PQ or HLG transfer function."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_transfer",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() in HDR_TRANSFERS
    except subprocess.CalledProcessError:
        return False


def is_portrait_source(video: Path) -> bool:
    """Return True if the video's height > width (portrait / vertical)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        )
        w, h = map(int, out.stdout.strip().split(","))
        return h > w
    except Exception:
        return False


# -------- Per-segment extraction (Rule 2 + Rule 3) --------------------------


def extract_segment(
    source: Path,
    seg_start: float,
    duration: float,
    grade_filter: str,
    out_path: Path,
    preview: bool = False,
    draft: bool = False,
    vertical: bool = False,
    vertical_center: bool = False,
    vertical_face: bool = False,
    vertical_gentle: bool = False,
    force_cpu: bool = False,
    prefer_stream_copy: bool = False,
) -> None:
    """Extract a cut range as its own MP4 with grade + 30ms audio fades baked in.

    `-ss` before `-i` for fast accurate seeking. Scale to 1080p from 4K.
    Portrait sources (height > width) are scaled by height to preserve orientation.
    `--vertical` / `--vertical-gentle` — 1080×1920 with slow subtle face drift (default for shorts).
    `--vertical-center` — static center crop, no face tracking.
    `--vertical-face` — responsive face-tracking pan.

    Quality ladder:
      - final (default): 1080p libx264 fast CRF 20
      - preview:         1080p libx264 medium CRF 22 (evaluable for QC)
      - draft:           720p libx264 ultrafast CRF 28 (cut-point check only)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = "draft" if draft else ("preview" if preview else "final")

    if vertical_face:
        from vertical_face import reframe_video

        reframe_video(
            source,
            out_path,
            start=seg_start,
            duration=duration,
            mode="full",
            profile=profile,
            force_cpu=force_cpu,
        )
        return
    if vertical or vertical_gentle:
        from vertical_face import reframe_video

        reframe_video(
            source,
            out_path,
            start=seg_start,
            duration=duration,
            mode="gentle",
            profile=profile,
            force_cpu=force_cpu,
        )
        return
    if vertical_center:
        scale = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    else:
        portrait = is_portrait_source(source)
        if draft:
            scale = "scale=-2:1280" if portrait else "scale=1280:-2"
        else:
            scale = "scale=-2:1920" if portrait else "scale=1920:-2"

    vf_parts: list[str] = []
    if is_hdr_source(source):
        vf_parts.append(TONEMAP_CHAIN)
    vf_parts.append(scale)
    if grade_filter:
        vf_parts.append(grade_filter)
    vf = ",".join(vf_parts)

    # 30ms audio fades at both edges (Rule 3) — prevent pops
    fade_out_start = max(0.0, duration - 0.03)
    af = f"afade=t=in:st=0:d=0.03,afade=t=out:st={fade_out_start:.3f}:d=0.03"

    if prefer_stream_copy and try_stream_copy_extract(source, seg_start, duration, out_path):
        return

    venc = list(video_encode_args(profile, force_cpu))
    common = [
        "ffmpeg", "-y",
        "-fflags", "+discardcorrupt+genpts",
        "-err_detect", "ignore_err",
        "-ss", f"{seg_start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", vf,
    ]

    def _attempt(*, cpu: bool, with_af: bool, with_audio: bool) -> None:
        enc = list(video_encode_args(profile, force_cpu=cpu))
        cmd = common + (["-af", af] if with_af else []) + enc + ["-r", "24"]
        if with_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        else:
            cmd += ["-an"]
        cmd += ["-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    last_err: subprocess.CalledProcessError | None = None
    cpus = (True,) if force_cpu else (False, True)
    for cpu in cpus:
        for with_af in (True, False):
            for with_audio in (True, False):
                try:
                    _attempt(cpu=cpu, with_af=with_af, with_audio=with_audio)
                    return
                except subprocess.CalledProcessError as exc:
                    last_err = exc
    if last_err:
        raise last_err


def extract_all_segments(
    edl: dict,
    edit_dir: Path,
    preview: bool,
    draft: bool = False,
    vertical: bool = False,
    vertical_center: bool = False,
    vertical_face: bool = False,
    vertical_gentle: bool = False,
    force_cpu: bool = False,
) -> list[Path]:
    """Extract every EDL range into edit_dir/clips_graded/seg_NN.mp4.
    Returns the ordered list of segment paths.

    If the EDL `grade` is "auto", analyze each segment range with
    `auto_grade_for_clip` and apply a per-segment subtle correction.
    Otherwise, apply the same preset/raw filter to every segment.
    """
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    clips_dir = edit_dir / (
        "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]

    seg_paths: list[Path] = []
    print(f"extracting {len(ranges)} segment(s) → {clips_dir.name}/")
    if is_auto:
        print("  (auto-grade per segment: analyzing each range)")
    for i, r in enumerate(ranges):
        src_name = r["source"]
        src_path = resolve_path(sources[src_name], edit_dir)
        start = float(r["start"])
        end = float(r["end"])
        duration = end - start
        out_path = clips_dir / f"seg_{i:02d}_{src_name}.mp4"

        if is_auto:
            seg_filter, _stats = auto_grade_for_clip(src_path, start=start, duration=duration, verbose=False)
        else:
            seg_filter = resolved

        note = r.get("beat") or r.get("note") or ""
        print(f"  [{i:02d}] {src_name}  {start:7.2f}-{end:7.2f}  ({duration:5.2f}s)  {note}")
        if is_auto:
            print(f"        grade: {seg_filter or '(none)'}")
        prefer_copy = should_stream_copy_first(duration, r)
        if prefer_copy:
            print("        extract: stream copy + audio (long B-roll)")
        extract_segment(
            src_path, start, duration, seg_filter, out_path,
            preview=preview, draft=draft, vertical=vertical,
            vertical_center=vertical_center, vertical_face=vertical_face,
            vertical_gentle=vertical_gentle, force_cpu=force_cpu,
            prefer_stream_copy=prefer_copy,
        )
        seg_paths.append(out_path)

    return seg_paths


# -------- Lossless concat ----------------------------------------------------


def concat_segments_scaled(
    segment_paths: list[Path],
    out_path: Path,
    *,
    profile: str = "final",
    force_cpu: bool = False,
) -> None:
    """One 1080p encode after concat when segments differ in resolution (copy + re-encode mix)."""
    n = len(segment_paths)
    cmd: list[str] = ["ffmpeg", "-y"]
    for p in segment_paths:
        cmd += ["-i", str(p)]
    v_parts: list[str] = []
    a_parts: list[str] = []
    concat_in: list[str] = []
    for i in range(n):
        v_parts.append(
            f"[{i}:v]scale=1920:-2:flags=fast_bilinear,setsar=1,format=yuv420p[v{i}]"
        )
        if probe_has_audio(segment_paths[i]):
            a_parts.append(
                f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]"
            )
        else:
            d = probe_duration(segment_paths[i])
            a_parts.append(
                f"anullsrc=duration={d:.3f}:sample_rate=48000:channel_layout=stereo[a{i}]"
            )
        concat_in.append(f"[v{i}][a{i}]")
    fc = ";".join(v_parts + a_parts) + f";{''.join(concat_in)}concat=n={n}:v=1:a=1[v][a]"
    last_err: subprocess.CalledProcessError | None = None
    for cpu in (force_cpu, True):
        try:
            subprocess.run(
                cmd
                + [
                    "-filter_complex", fc,
                    "-map", "[v]", "-map", "[a]",
                    *video_encode_args(profile, force_cpu=cpu),
                    "-r", "24",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart",
                    str(out_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return
        except subprocess.CalledProcessError as exc:
            last_err = exc
    if last_err:
        raise last_err


def concat_segments(
    segment_paths: list[Path],
    out_path: Path,
    edit_dir: Path,
    *,
    profile: str = "final",
    force_cpu: bool = False,
) -> None:
    """Concat segments; copy when uniform, else one scaled re-encode."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = {probe_video_size(p) for p in segment_paths}
    if len(sizes) > 1:
        print(f"concat (scale→1080p) → {out_path.name}")
        concat_segments_scaled(segment_paths, out_path, profile=profile, force_cpu=force_cpu)
        return

    concat_list = edit_dir / "_concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"concat → {out_path.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print(f"concat fallback (scale→1080p) → {out_path.name}")
        concat_segments_scaled(segment_paths, out_path, profile=profile, force_cpu=force_cpu)
    finally:
        concat_list.unlink(missing_ok=True)


# -------- Master SRT (Rule 5) ------------------------------------------------


PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws = w.get("start")
        we = w.get("end")
        if ws is None or we is None:
            continue
        if we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def build_master_srt(edl: dict, edit_dir: Path, out_path: Path) -> None:
    """Build an output-timeline SRT from per-source transcripts.

    - 2-word chunks (break on any punctuation in between)
    - UPPERCASE text
    - Output times computed as word.start - segment_start + segment_offset
    """
    project_dir = resolve_project_dir(edit_dir)
    transcripts_dir = resolve_transcripts_dir(project_dir)
    sources = edl["sources"]

    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0

    for r in edl["ranges"]:
        src_name = r["source"]
        seg_start = float(r["start"])
        seg_end = float(r["end"])
        seg_duration = seg_end - seg_start

        tr_path = transcripts_dir / f"{src_name}.json"
        if not tr_path.exists():
            print(f"  no transcript for {src_name}, skipping captions for this segment")
            seg_offset += seg_duration
            continue

        transcript = json.loads(tr_path.read_text())
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)

        # Group into 2-word chunks, break on punctuation
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
            # Break if the current text ends in punctuation or we hit 2 words
            ends_in_punct = bool(text) and text[-1] in PUNCT_BREAK
            if len(current) >= 2 or ends_in_punct:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        for chunk in chunks:
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip()
            # Strip trailing punctuation for cleaner uppercase look
            text = text.rstrip(",;:")
            text = text.upper()
            entries.append((out_start, out_end, text))

        seg_offset += seg_duration

    # Sort and write as SRT
    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}")
        lines.append(t)
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"master SRT → {out_path.name} ({len(entries)} cues)")


# -------- Loudness normalization (social-ready audio) -----------------------


# Social-media standard: -14 LUFS integrated, -1 dBTP peak, LRA 11 LU.
# Matches YouTube / Instagram / TikTok / X / LinkedIn normalization targets.
LOUDNORM_I = -14.0
LOUDNORM_TP = -1.0
LOUDNORM_LRA = 11.0


def measure_loudness(video_path: Path) -> dict[str, str] | None:
    """Run ffmpeg loudnorm first pass and parse the JSON measurement.

    Returns a dict with measured_i, measured_tp, measured_lra, measured_thresh,
    target_offset, or None if measurement failed.
    """
    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(video_path),
        "-af", filter_str,
        "-vn", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # loudnorm prints the JSON to stderr at the end of the run
    stderr = proc.stderr

    # Find the JSON block — loudnorm output contains a `{ ... }` block
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stderr[start : end + 1])
    except json.JSONDecodeError:
        return None
    needed = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    if not needed.issubset(data.keys()):
        return None
    return data


def apply_loudnorm_two_pass(
    input_path: Path,
    output_path: Path,
    preview: bool = False,
) -> bool:
    """Run two-pass loudnorm on input_path, write normalized copy to output_path.

    Returns True on success, False if measurement failed (caller should fall
    back to copying the input unchanged).

    In preview mode, skips the measurement pass and uses a one-pass approximation
    for speed. Final mode always does the proper two-pass.
    """
    if preview:
        # One-pass approximation — faster, slightly less accurate.
        filter_str = f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(input_path),
            "-c:v", "copy",
            "-af", filter_str,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            str(output_path),
        ]
        print(f"  loudnorm (1-pass preview) → {output_path.name}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True

    # Full two-pass
    print(f"  loudnorm pass 1: measuring {input_path.name}")
    measurement = measure_loudness(input_path)
    if measurement is None:
        print("  loudnorm measurement failed — falling back to 1-pass")
        return apply_loudnorm_two_pass(input_path, output_path, preview=True)

    print(f"    measured: I={measurement['input_i']} LUFS  "
          f"TP={measurement['input_tp']}  LRA={measurement['input_lra']}")

    filter_str = (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={measurement['input_i']}"
        f":measured_TP={measurement['input_tp']}"
        f":measured_LRA={measurement['input_lra']}"
        f":measured_thresh={measurement['input_thresh']}"
        f":offset={measurement['target_offset']}"
        f":linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", str(input_path),
        "-c:v", "copy",
        "-af", filter_str,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    print(f"  loudnorm pass 2: normalizing → {output_path.name}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return True


# -------- Final compositing (Rule 1 + Rule 4) -------------------------------


def build_final_composite(
    base_path: Path,
    overlays: list[dict],
    subtitles_path: Path | None,
    out_path: Path,
    edit_dir: Path,
    subtitle_style: str | None = None,
    force_cpu: bool = False,
) -> None:
    """Final pass: base → overlays (PTS-shifted) → subtitles LAST → out.

    If there are no overlays and no subtitles, just copy base to out.
    """
    has_overlays = bool(overlays)
    has_subs = subtitles_path is not None and subtitles_path.exists()

    if not has_overlays and not has_subs:
        # Nothing to do — just rename/copy base to final name
        run(["ffmpeg", "-y", "-i", str(base_path), "-c", "copy", str(out_path)], quiet=True)
        return

    # Subtitles-only: use -vf (filter_complex breaks subtitle paths on Windows)
    if has_subs and not has_overlays:
        style = subtitle_style or SUB_FORCE_STYLE
        subs_cwd, subs_name = stage_srt_for_ffmpeg(subtitles_path)
        vf = f"subtitles={subs_name}:force_style='{style}'"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(base_path),
            "-vf", vf,
            "-map", "0:v", "-map", "0:a",
            *video_encode_args("high", force_cpu),
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        print(f"compositing → {out_path.name}")
        print(f"  overlays: 0, subtitles: yes")
        subprocess.run(cmd, check=True, cwd=subs_cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return

    inputs: list[str] = ["-i", str(base_path)]
    for ov in overlays:
        ov_path = resolve_path(ov["file"], edit_dir)
        inputs += ["-i", str(ov_path)]

    filter_parts: list[str] = []
    # PTS-shift every overlay so its frame 0 lands at start_in_output.
    # HyperFrames WebM overlays carry VP9 alpha — convert to yuva420p and
    # composite with alpha=premultiplied or the full frame reads as opaque black.
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        filter_parts.append(
            f"[{idx}:v]format=yuva420p,setpts=PTS-STARTPTS+{t}/TB[ov{idx}]"
        )

    # Chain overlays on top of base
    current = "[0:v]"
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"])
        dur = float(ov["duration"])
        end = t + dur
        next_label = f"[v{idx}]"
        filter_parts.append(
            f"{current}[ov{idx}]overlay=format=auto:alpha=premultiplied:"
            f"enable='between(t,{t:.3f},{end:.3f})'{next_label}"
        )
        current = next_label

    # Subtitles LAST — Rule 1
    if has_subs:
        style = subtitle_style or SUB_FORCE_STYLE
        subs_cwd, subs_name = stage_srt_for_ffmpeg(subtitles_path)
        filter_parts.append(
            f"{current}subtitles={subs_name}:force_style='{style}'[outv]"
        )
        out_label = "[outv]"
    else:
        # Rename the last overlay output to [outv] for consistency
        if has_overlays:
            filter_parts.append(f"{current}null[outv]")
            out_label = "[outv]"
        else:
            out_label = "[0:v]"

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", out_label,
        "-map", "0:a",
        *video_encode_args("high", force_cpu),
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]
    print(f"compositing → {out_path.name}")
    print(f"  overlays: {len(overlays)}, subtitles: {'yes' if has_subs else 'no'}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output video path")
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: 1080p, medium, CRF 22 — evaluable for QC, faster than final.",
    )
    ap.add_argument(
        "--draft",
        action="store_true",
        help="Draft mode: 720p, ultrafast, CRF 28 — cut-point verification only.",
    )
    ap.add_argument(
        "--build-subtitles",
        action="store_true",
        help="Build master.srt from transcripts + EDL offsets before compositing",
    )
    ap.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Skip subtitles even if the EDL references one",
    )
    ap.add_argument(
        "--no-loudnorm",
        action="store_true",
        help="Skip audio loudness normalization. Default is on (-14 LUFS, -1 dBTP, LRA 11).",
    )
    ap.add_argument(
        "--vertical",
        action="store_true",
        help="Output 1080×1920 (9:16) with gentle face drift — default for vertical shorts.",
    )
    ap.add_argument(
        "--vertical-center",
        action="store_true",
        help="Output 1080×1920 (9:16) via static center crop (no face drift).",
    )
    ap.add_argument(
        "--vertical-gentle",
        action="store_true",
        help="Alias for --vertical (gentle face drift).",
    )
    ap.add_argument(
        "--vertical-face",
        action="store_true",
        help="Output 1080×1920 (9:16) with responsive face-tracking pan.",
    )
    ap.add_argument(
        "--cpu-encode",
        action="store_true",
        help="Force libx264 CPU encoding instead of GPU (h264_amf when available)",
    )
    args = ap.parse_args()

    vertical_modes = sum(
        [
            args.vertical or args.vertical_gentle,
            args.vertical_center,
            args.vertical_face,
        ]
    )
    if vertical_modes > 1:
        sys.exit("use only one vertical mode: --vertical, --vertical-center, or --vertical-face")

    print(f"video encoder: {encoder_label(args.cpu_encode)}")

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")

    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    project_dir = resolve_project_dir(edit_dir)
    out_path = args.output.resolve()
    is_vertical = args.vertical or args.vertical_gentle or args.vertical_center or args.vertical_face
    sub_style = VERTICAL_SUB_FORCE_STYLE if is_vertical else SUB_FORCE_STYLE

    use_gentle = args.vertical or args.vertical_gentle
    segment_paths = extract_all_segments(
        edl, edit_dir, preview=args.preview, draft=args.draft,
        vertical=use_gentle, vertical_center=args.vertical_center,
        vertical_face=args.vertical_face, vertical_gentle=use_gentle,
        force_cpu=args.cpu_encode,
    )

    # 2. Concat → base
    if args.draft:
        base_name = "base_draft.mp4"
    elif args.preview:
        base_name = "base_preview.mp4"
    else:
        base_name = "base.mp4"
    base_path = edit_dir / base_name
    profile = "draft" if args.draft else ("preview" if args.preview else "final")
    concat_segments(
        segment_paths, base_path, edit_dir,
        profile=profile, force_cpu=args.cpu_encode,
    )

    # 3. Subtitles: build if requested, resolve final path
    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, edit_dir, subs_path)
        elif edl.get("subtitles"):
            subs_path = resolve_path(edl["subtitles"], edit_dir)
            if not subs_path.exists():
                print(f"warning: subtitles path in EDL does not exist: {subs_path}")
                subs_path = None

    # 4. Composite (overlays + subtitles LAST) → intermediate (pre-loudnorm) path
    overlays = edl.get("overlays") or []
    if args.no_loudnorm:
        build_final_composite(
            base_path, overlays, subs_path, out_path, project_dir,
            subtitle_style=sub_style, force_cpu=args.cpu_encode,
        )
    else:
        tmp_composite = out_path.with_suffix(".prenorm.mp4")
        build_final_composite(
            base_path, overlays, subs_path, tmp_composite, project_dir,
            subtitle_style=sub_style, force_cpu=args.cpu_encode,
        )
        print("loudness normalization → social-ready (-14 LUFS / -1 dBTP / LRA 11)")
        apply_loudnorm_two_pass(tmp_composite, out_path, preview=args.draft)
        tmp_composite.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
