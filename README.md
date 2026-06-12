# Video Clippy

Turn long-form talking-head footage into **3+ minute YouTube clips** — transcribed with local Whisper, cut on phrase boundaries, rendered at 1080p with an optional outro.

Built on the [video-use](https://github.com/browser-use/video-use) editing helpers.

## Features

- **Local transcription** — faster-whisper, no API key required
- **Batch pipeline** — drop MP4s in `footage/`, get `(Clips)` folders back
- **Continuous takes** — one uninterrupted span per clip (no filler-word micro-cuts)
- **Outro compositing** — fade body out + loudness-matched outro
- **Docker** — portable CPU render environment
- **CLI** — `video-clippy batch`, `render`, `transcribe`, `pack`

## Requirements

- **ffmpeg** ≥ 4.x (with libx264 + libass for subtitles)
- **Python** ≥ 3.10
- ~4 GB disk for Whisper `small.en` model (downloaded on first run)

Optional: AMD GPU on Windows/Linux (`h264_amf`) for faster encodes — auto-detected, or set `VIDEO_USE_GPU=0` to force CPU.

## Quick start

```bash
git clone https://github.com/ForrestHODL/video-clippy.git
cd video-clippy

pip install -e .

# Drop source videos in footage/ and add an outro (optional)
# footage/my-interview.mp4
# footage/outro.mp4

video-clippy batch --list    # show queue
video-clippy batch           # transcribe → plan → render → export
```

Deliverables appear in `footage/<Title> (Clips)/`. Project files (transcripts, EDLs) land in `footage/edit/<slug>/`.

## Docker

```bash
docker compose build
docker compose run --rm clippy batch --list
docker compose run --rm clippy batch
```

Mount your footage directory (configured in `docker-compose.yml`). Whisper models cache in the `clippy-cache` volume.

## CLI reference

| Command | Description |
|---------|-------------|
| `video-clippy batch` | Process all `footage/*.mp4` (skips `outro.mp4`) |
| `video-clippy batch --list` | Show pending/done queue |
| `video-clippy batch --only <slug>` | Process one video |
| `video-clippy batch --force` | Re-render even if done |
| `video-clippy render <edl.json> -o out.mp4` | Render a single EDL |
| `video-clippy transcribe [path]` | Transcribe footage directory |
| `video-clippy pack --edit-dir <dir>` | Build `takes_packed.md` from transcripts |

Set `VIDEO_CLIPPY_ROOT` if running from outside the repo root.

## Project layout

```
video-clippy/
├── studio/              # CLI + batch pipeline
│   └── pipeline/        # transcribe, plan clips, render
├── video-use/helpers/   # render, grade, whisper, horizontal_clips
├── footage/             # your source videos (gitignored)
│   ├── outro.mp4        # optional end card
│   ├── edit/            # per-project transcripts & EDLs
│   └── .studio/         # batch state (resumable)
├── Dockerfile
└── pyproject.toml
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_CLIPPY_ROOT` | auto-detect | Project root path |
| `VIDEO_USE_GPU` | `auto` | `0` = force libx264 CPU encode |
| `ELEVENLABS_API_KEY` | — | Optional cloud transcription |

## How it works

1. **Transcribe** — chunk-based faster-whisper with phrase-level word timings
2. **Plan** — split into ~3–6 min continuous spans on phrase boundaries (min 3 min per clip)
3. **Render** — EDL → segment extract → concat → optional outro composite
4. **Export** — copy finals to `footage/<Title> (Clips)/`

Batch progress is saved in `footage/.studio/batch-state.json` — safe to stop and resume.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Video-use helpers are MIT-licensed from [browser-use/video-use](https://github.com/browser-use/video-use).
