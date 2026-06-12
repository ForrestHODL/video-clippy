# Contributing

Thanks for helping improve Video Clippy!

## Development setup

```bash
git clone https://github.com/ForrestHODL/video-clippy.git
cd video-clippy
pip install -e .
```

Ensure `ffmpeg` is on your PATH.

## Running the batch pipeline

```bash
video-clippy batch --list
video-clippy batch --only my-slug
```

## Code layout

- `studio/` — packaged CLI and batch orchestration
- `video-use/helpers/` — low-level render, transcribe, and clip-building tools
- `footage/` — runtime data only (not committed)

## Pull requests

- Keep changes focused
- Test with a short sample clip before submitting render pipeline changes
- Do not commit footage, API keys, or generated edit outputs
