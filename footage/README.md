# Footage

Drop source videos here (`.mp4`, `.mov`, `.mkv`).

```
footage/
├── my-interview.mp4       ← source
├── outro.mp4              ← optional end card (skipped by batch)
├── edit/                  ← generated per-project (transcripts, EDLs, clips)
├── .studio/               ← batch state (resumable)
└── My Interview (Clips)/  ← exported finals
```

Run the batch:

```bash
video-clippy batch
```
