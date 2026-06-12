# Video Clippy — CPU render + local Whisper transcription
# Build:  docker build -t video-clippy .
# Run:    docker compose run --rm clippy batch --list

FROM python:3.12-bookworm

LABEL org.opencontainers.image.title="Video Clippy"
LABEL org.opencontainers.image.description="Transcribe, cut, and render YouTube clips from raw footage"

ENV VIDEO_CLIPPY_ROOT=/studio \
    VIDEO_USE_GPU=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/studio/.cache/huggingface \
    XDG_CACHE_HOME=/studio/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libass9 \
    libsndfile1 \
    curl \
    ca-certificates \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

WORKDIR /studio

COPY requirements.txt pyproject.toml ./
COPY studio/ ./studio/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY video-use/helpers/ ./video-use/helpers/
COPY footage/README.md ./footage/
RUN mkdir -p footage/edit footage/.studio footage/.cache

VOLUME ["/studio/footage"]

ENTRYPOINT ["video-clippy"]
CMD ["batch", "--list"]
