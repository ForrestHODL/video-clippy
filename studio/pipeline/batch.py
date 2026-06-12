"""Batch-process every .mp4 in footage/ — continuous clips, 1080p + outro.

Resumable via footage/.studio/batch-state.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from studio.paths import edit_dir, footage_dir, helpers_dir, studio_root
from studio.pipeline.chunk_transcribe import chunk_transcribe
from studio.pipeline.toni_template import write_project

ROOT = studio_root()
FOOTAGE = footage_dir()
EDIT = edit_dir()
STATE_PATH = FOOTAGE / ".studio" / "batch-state.json"
SKIP_FILES = {"outro.mp4"}


def env() -> dict:
    e = os.environ.copy()
    e.setdefault("PYTHONIOENCODING", "utf-8")
    e["PYTHONPATH"] = str(helpers_dir()) + os.pathsep + e.get("PYTHONPATH", "")
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Links"
    if winget.is_dir():
        e["PATH"] = str(winget) + os.pathsep + e.get("PATH", "")
    return e


def slug_from_stem(stem: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return s[:72] or "clip"


def source_key_from_stem(stem: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return s[:80] or "source"


def deliverable_from_stem(stem: str) -> str:
    t = re.sub(r"\s+", " ", stem).strip()
    t = re.sub(r"\s*(vanlife|van life|tour|documentary|vl)\s*$", "", t, flags=re.I).strip()
    t = re.sub(r"^(.{60}).*", r"\1", t)
    if len(t) < 8:
        t = stem[:60]
    return t


def probe_duration(video: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"videos": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def discover_videos() -> list[dict]:
    items: list[dict] = []
    for mp4 in sorted(FOOTAGE.glob("*.mp4")):
        if mp4.name in SKIP_FILES:
            continue
        stem = mp4.stem
        slug = slug_from_stem(stem)
        items.append({
            "slug": slug,
            "file": mp4.name,
            "deliverable": deliverable_from_stem(stem),
            "source_key": source_key_from_stem(stem),
            "path": str(mp4),
        })
    return items


def deliverable_dir(proj: dict) -> Path:
    return FOOTAGE / f"{proj['deliverable']} (Clips)"


def is_done(proj: dict, state: dict) -> bool:
    rec = state.get("videos", {}).get(proj["slug"], {})
    if rec.get("status") == "done":
        d = deliverable_dir(proj)
        if d.is_dir() and any(d.glob("*.mp4")):
            return True
    return False


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env())


def plan_clips(phrases: list[dict], duration: float, min_clips: int = 3) -> list[dict]:
    if not phrases:
        return []
    usable = [p for p in phrases if p["end"] - p["start"] >= 0.3]
    if not usable:
        return []
    target = max(min_clips, int(duration / 240) + 1)
    target = min(target, 6)
    chunk_size = max(1, len(usable) // target)
    clips: list[dict] = []
    i = 0
    n = 1
    while i < len(usable):
        group = usable[i : i + chunk_size]
        if not group:
            break
        if i + chunk_size >= len(usable):
            group = usable[i:]
        start = group[0]["start"]
        end = group[-1]["end"]
        if duration - end < 60 and clips:
            clips[-1]["segments"][0]["end_seconds"] = duration - 0.5
            break
        title_words = re.sub(r"[^a-zA-Z0-9 ]", "", group[0]["text"]).split()[:8]
        title = " ".join(title_words).title()
        if len(title) < 8:
            title = f"Part {n}"
        clips.append({
            "id": f"{n:02d}_clip",
            "title": title[:80],
            "segments": [{
                "start_seconds": max(0, start - 0.22),
                "end_seconds": min(duration, end + 0.35),
                "quote": group[0]["text"][:120],
            }],
        })
        if i + chunk_size >= len(usable):
            break
        i += chunk_size
        n += 1
    if clips:
        last = clips[-1]
        d = last["segments"][0]["end_seconds"] - last["segments"][0]["start_seconds"]
        if d < 180 and len(clips) > 1:
            prev = clips[-2]
            prev["segments"][0]["end_seconds"] = last["segments"][0]["end_seconds"]
            clips.pop()
    if clips and (clips[0]["segments"][0]["end_seconds"] - clips[0]["segments"][0]["start_seconds"]) < 180 and len(clips) > 1:
        clips[1]["segments"][0]["start_seconds"] = clips[0]["segments"][0]["start_seconds"]
        clips.pop(0)
    return [c for c in clips if c["segments"][0]["end_seconds"] - c["segments"][0]["start_seconds"] >= 180]


def refine_titles(clips: list[dict], phrases: list[dict]) -> list[dict]:
    for c in clips:
        s = c["segments"][0]["start_seconds"]
        e = c["segments"][0]["end_seconds"]
        mid = [p for p in phrases if p["start"] >= s and p["end"] <= e + 1]
        if mid:
            words = re.sub(r"\s+", " ", mid[0]["text"]).strip()
            if len(words) > 15:
                c["title"] = (words[:70] + "…") if len(words) > 70 else words
    return clips


def process_one(proj: dict, state: dict) -> None:
    slug = proj["slug"]
    video = FOOTAGE / proj["file"]
    project_edit_dir = EDIT / slug
    project_edit_dir.mkdir(parents=True, exist_ok=True)
    dest = deliverable_dir(proj)

    print(f"\n{'=' * 60}\n{slug}\n  {proj['file']}", flush=True)
    if not video.exists():
        print("  MISSING source — skip", flush=True)
        state["videos"][slug] = {"status": "missing", "file": proj["file"]}
        save_state(state)
        return

    t0 = time.time()
    dur = probe_duration(video)
    print(f"  duration {dur/60:.1f} min", flush=True)

    state["videos"][slug] = {
        "status": "processing",
        "file": proj["file"],
        "deliverable": proj["deliverable"],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_state(state)

    tpath = project_edit_dir / "transcripts" / f"{video.stem}.json"
    if not tpath.exists():
        print("  transcribing…", flush=True)
        chunk_transcribe(video, project_edit_dir)
    else:
        print("  transcript cached", flush=True)

    run([
        sys.executable, str(helpers_dir() / "pack_transcripts.py"),
        "--edit-dir", str(project_edit_dir),
    ])

    if not tpath.exists():
        tpath = next(project_edit_dir.glob("transcripts/*.json"))
    data = json.loads(tpath.read_text(encoding="utf-8"))
    words = data.get("words") or []

    sys.path.insert(0, str(helpers_dir()))
    from horizontal_clips import parse_phrases  # noqa: E402

    phrases = parse_phrases(words)
    clips = refine_titles(plan_clips(phrases, dur), phrases)
    if not clips:
        print("  NO CLIPS — skip", flush=True)
        state["videos"][slug]["status"] = "no_clips"
        save_state(state)
        return

    (project_edit_dir / "clips.json").write_text(json.dumps(clips, indent=2), encoding="utf-8")
    write_project(project_edit_dir, video, proj["source_key"], dur, clips, proj["deliverable"])
    print(f"  {len(clips)} clips planned", flush=True)

    run([sys.executable, str(project_edit_dir / "build_clips.py"), "--validate-only"])
    run([sys.executable, str(project_edit_dir / "build_clips.py"), "--final", "--force"])

    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.mp4"):
        old.unlink()
    exported = 0
    for mp4 in sorted((project_edit_dir / "clips").glob("*.mp4")):
        if "_body" in mp4.name:
            continue
        shutil.copy2(mp4, dest / mp4.name)
        exported += 1

    elapsed = time.time() - t0
    state["videos"][slug] = {
        "status": "done",
        "file": proj["file"],
        "deliverable": proj["deliverable"],
        "clips": exported,
        "duration_min": round(dur / 60, 1),
        "elapsed_min": round(elapsed / 60, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_state(state)
    print(f"  exported {exported} clips → {dest.name} ({elapsed/60:.1f} min)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch-process footage into YouTube clips")
    ap.add_argument("--list", action="store_true", help="Show queue and exit")
    ap.add_argument("--only", metavar="SLUG", help="Process one slug only")
    ap.add_argument("--force", action="store_true", help="Re-process even if done")
    args = ap.parse_args(argv)

    videos = discover_videos()
    state = load_state()

    if args.list:
        for v in videos:
            done = is_done(v, state)
            mark = "done" if done else "pending"
            print(f"  [{mark}] {v['slug'][:50]:50}  {v['file'][:60]}")
        done_n = sum(1 for v in videos if is_done(v, state))
        print(f"\n{len(videos)} total, {done_n} done, {len(videos) - done_n} pending")
        return 0

    pending = []
    for v in videos:
        if args.only and v["slug"] != args.only and not v["slug"].startswith(args.only):
            continue
        if not args.force and is_done(v, state):
            print(f"skip (done): {v['slug']}", flush=True)
            continue
        pending.append(v)

    print(f"Processing {len(pending)} video(s)…", flush=True)
    for i, proj in enumerate(pending, 1):
        print(f"\n>>> [{i}/{len(pending)}]", flush=True)
        try:
            process_one(proj, state)
        except subprocess.CalledProcessError as e:
            state["videos"][proj["slug"]] = {
                "status": "error",
                "file": proj["file"],
                "error": str(e),
                "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_state(state)
            print(f"  ERROR: {e} — continuing to next video", flush=True)

    print("\nBATCH COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
