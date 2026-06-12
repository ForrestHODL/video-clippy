"""Video Clippy — command-line entry point."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from studio.paths import edit_dir, footage_dir, helpers_dir, studio_root


def _py() -> str:
    return sys.executable


def _run(script: Path, *args: str) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("VIDEO_CLIPPY_ROOT", str(studio_root()))
    env["PYTHONPATH"] = str(helpers_dir()) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call([_py(), str(script), *args], env=env)


def cmd_batch(args: argparse.Namespace) -> int:
    from studio.pipeline.batch import main as batch_main

    argv: list[str] = []
    if args.list:
        argv.append("--list")
    if args.only:
        argv.extend(["--only", args.only])
    if args.force:
        argv.append("--force")
    return batch_main(argv)


def cmd_render(args: argparse.Namespace) -> int:
    return _run(helpers_dir() / "render.py", args.edl, "-o", args.output, *args.extra)


def cmd_transcribe(args: argparse.Namespace) -> int:
    target = args.path or str(footage_dir())
    return _run(helpers_dir() / "transcribe_batch.py", target, *args.extra)


def cmd_pack(args: argparse.Namespace) -> int:
    target = args.edit_dir or str(edit_dir())
    return _run(helpers_dir() / "pack_transcripts.py", "--edit-dir", target, *args.extra)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="video-clippy",
        description="Video Clippy — transcribe, cut, and render clips",
    )
    parser.add_argument(
        "--root",
        help="Project root (default: auto-detect or VIDEO_CLIPPY_ROOT)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="Batch-process footage/*.mp4 into YouTube clips")
    batch.add_argument("--list", action="store_true", help="Show queue")
    batch.add_argument("--only", metavar="SLUG", help="Process one slug")
    batch.add_argument("--force", action="store_true", help="Re-render even if done")
    batch.set_defaults(func=cmd_batch)

    render = sub.add_parser("render", help="Render an EDL to MP4")
    render.add_argument("edl", help="Path to edl.json")
    render.add_argument("-o", "--output", required=True, help="Output MP4 path")
    render.add_argument("extra", nargs=argparse.REMAINDER, help="Extra render.py flags")
    render.set_defaults(func=cmd_render)

    transcribe = sub.add_parser("transcribe", help="Transcribe footage with Whisper")
    transcribe.add_argument("path", nargs="?", help="Footage file or directory")
    transcribe.add_argument("extra", nargs=argparse.REMAINDER, help="Extra transcribe_batch.py flags")
    transcribe.set_defaults(func=cmd_transcribe)

    pack = sub.add_parser("pack", help="Pack transcript JSON into takes_packed.md")
    pack.add_argument("--edit-dir", help="Edit project directory")
    pack.add_argument("extra", nargs=argparse.REMAINDER, help="Extra pack_transcripts.py flags")
    pack.set_defaults(func=cmd_pack)

    args = parser.parse_args(argv)
    if args.root:
        os.environ["VIDEO_CLIPPY_ROOT"] = str(Path(args.root).resolve())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
