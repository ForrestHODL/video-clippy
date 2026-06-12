# Video Clippy

Turn long-form talking-head footage into **3+ minute YouTube clips** — transcribed with local Whisper, cut on phrase boundaries, rendered at 1080p with an optional outro.

Video Clippy wraps and extends two open-source projects:

| Project | What it provides | Repo |
|---------|------------------|------|
| **[video-use](https://github.com/browser-use/video-use)** | Transcription, EDL rendering, grading, horizontal clip engine, outro compositing | [browser-use/video-use](https://github.com/browser-use/video-use) |
| **[HyperFrames](https://github.com/heygen-com/hyperframes)** | HTML/CSS motion graphics — title cards, lower thirds, animated overlays (optional) | [heygen-com/hyperframes](https://github.com/heygen-com/hyperframes) |

The batch pipeline and CLI in this repo sit on top of video-use. HyperFrames is not bundled here, but pairs well when you want AI-assisted motion graphics on top of your clips.

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

## Using with AI (Cursor, Claude, ChatGPT, etc.)

Video Clippy is designed to be driven by conversation. An AI agent reads your transcripts and EDLs, runs the CLI, and iterates on clip boundaries — you review, give feedback, it re-renders.

### Setup

1. Clone this repo and `pip install -e .`
2. Install **ffmpeg** and (optionally) **faster-whisper** deps via the install step above
3. Open the folder in **[Cursor](https://cursor.com)** or any AI IDE that can run terminal commands
4. Drop raw `.mp4` files in `footage/` plus an optional `footage/outro.mp4`

### Automated batch (hands-off)

Ask the AI:

> Process everything in `footage/` into YouTube clips. Use `video-clippy batch`, 1080p finals with outro.

The agent runs:

```bash
video-clippy batch --list   # see queue
video-clippy batch          # transcribe → plan → render → export
```

Outputs land in `footage/<Title> (Clips)/`. Safe to stop mid-run — progress is in `footage/.studio/batch-state.json`.

### Guided editing (more control)

For finer cuts, work one video at a time:

1. **Transcribe**
   ```bash
   video-clippy transcribe footage/my-video.mp4
   video-clippy pack --edit-dir footage/edit/my-slug
   ```
2. **Read** `footage/edit/<slug>/takes_packed.md` — the AI uses this to understand what's in the video
3. **Tell the AI what you want**, e.g.:
   - *"Make 3 clips about the solar setup, kitchen tour, and intro. Each at least 3 minutes."*
   - *"Clip 2 should start right before 'came back to Canada' — check the transcript."*
   - *"Extend the last clip to finish the sentence at 8:24."*
4. The AI edits `footage/edit/<slug>/clips.json` (or `build_clips.py` anchors), validates, and renders:
   ```bash
   video-clippy batch --only my-slug
   # or per-project:
   python footage/edit/my-slug/build_clips.py --validate-only
   python footage/edit/my-slug/build_clips.py --final --force
   ```

### Tips for good AI results

- **Point at the transcript** — clip boundaries should land on phrase edges, never mid-word
- **One topic per clip** — ask for continuous takes, not jump cuts, unless you want topic compilation
- **Review drafts first** — `build_clips.py --draft` renders fast 720p previews before finals
- **Iterate in plain language** — *"clip 1 audio goes dead at 1:05, skip the corrupt section"* beats tweaking timestamps yourself
- **Motion graphics (optional)** — for lower thirds or title cards, install [HyperFrames](https://github.com/heygen-com/hyperframes) (`npx hyperframes`) and ask the AI to composite overlays via `video-use/helpers/render.py`

### Example Cursor prompt

```
I dropped 4 van tour videos in footage/. For each one:
1. Run video-clippy batch (skip any already done)
2. Each clip should be 3+ minutes, continuous takes on phrase boundaries
3. Put finals in the (Clips) folders with outro

If a clip title is ugly, rename from the transcript content.
Tell me when each video finishes.
```

## Acknowledgements

- **[browser-use/video-use](https://github.com/browser-use/video-use)** — conversation-driven editing engine, Whisper transcription helpers, EDL format, render pipeline. MIT License.
- **[heygen-com/hyperframes](https://github.com/heygen-com/hyperframes)** — HTML-based motion graphics for overlays and title cards. Used optionally alongside this project. See their repo for license terms.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). See [Acknowledgements](#acknowledgements) for upstream projects.
