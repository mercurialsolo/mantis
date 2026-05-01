# Recordings

Set `record_video: true` on `POST /v1/predict` and you get a polished, narratable walkthrough video at the end of the run.

## Request

```jsonc
{
  "detached": true,
  "micro": "plans/example/...json",
  "record_video": true,
  "video_format": "mp4",   // or "webm" | "gif"
  "video_fps":    8,        // 1–30, defaults to 5
  ...
}
```

## What you get

```
0:00 ─── 0:03   TITLE CARD
                Mantis CUA · <plan name> · tenant · run id

0:03 ─── 9:30   CAPTIONED RUN
                Step 1: Navigate to listings page          [OK]
                Step 2: Verify private-seller listings     [OK]
                Step 3: Click only an organic listing      [OK]
                ...
                with sky-blue overlays for every action

9:30 ─── 9:35   OUTRO CARD
                Run complete · 3 viable leads · 1 with phone
                17 steps · 569 s · cost $0.42
```

## Action overlays

Visible on top of the run footage. Each kind of agent action gets its own visual cue:

| Agent action | Overlay |
|---|---|
| `CLICK` / `DOUBLE_CLICK` | Sky-blue expanding ripple at (x, y), 0.6 s |
| `KEY_PRESS` (`Ctrl+S`, `Tab`, `Enter`, `alt+Left`, …) | Slate badge bottom-right with the chord, 1.5 s |
| `TYPE` (typed text) | "⌨ Typing: \"…\"" caption near the top, 1.8 s |
| `SCROLL` (`up`/`down`/`left`/`right`) | Sky-blue arrow at the matching screen edge, 0.8 s |
| `DRAG` | Trail line + moving head dot from start to end, 0.9 s |
| `WAIT`, `NAVIGATE`, `DONE` | No overlay |

These work for **any computer-use scenario** — browser, file manager, terminal, dialogs, anything visible on the Xvfb display. The agent emits actions in pixel coordinates; the overlay renders at those pixels regardless of what's painted there.

## Downloading

```bash
RUN_ID="20260428_…"

# Polished walkthrough (default)
curl -fsS -o demo.mp4 \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  "$ENDPOINT/v1/runs/$RUN_ID/video"

# Raw screencast without overlays
curl -fsS -o demo_raw.mp4 \
  -H "X-Mantis-Token: $MANTIS_API_TOKEN" \
  "$ENDPOINT/v1/runs/$RUN_ID/video?raw=1"
```

| Path | Returns |
|---|---|
| `GET /v1/runs/{id}/video` | Polished mp4 (preferred) → raw mp4 (fallback) → 404 |
| `GET /v1/runs/{id}/video?raw=1` | Raw mp4 only → 404 |

## Format choice

| Format | Encoder | Encode CPU | Typical 10-min run | Use |
|---|---|---|---|---|
| `mp4` | libx264 ultrafast CRF 28 | low | 30–80 MB | downloads, sharing |
| `webm` | libvpx-vp9 cpu-used 5 CRF 32 | medium | 25–60 MB | web embedding |
| `gif` | palettegen + paletteuse | high | 50–200 MB | docs, Slack, demos |

For the typical "send my customer a recording" workflow, **mp4 at 5 fps** is the sweet spot. `gif` only makes sense for short demos < 60 s — file size grows fast.

## Result metadata

`{"action":"result", ...}` includes a `video` block:

```jsonc
{
  "video": {
    "path":           "/.../recording.mp4",
    "polished_path":  "/.../recording_polished.mp4",
    "format":         "mp4",
    "duration_seconds": 567.3,
    "bytes":          31457280,
    "actions": {
      "clicks":  17,
      "keys":    3,
      "types":   2,
      "scrolls": 8,
      "drags":   0
    },
    "clicks": 17,    // backwards-compat field
    "error":  null
  }
}
```

`polished_path` is set only when the post-process compose step succeeded. If it failed (rare — usually means `libass` isn't built into the container's ffmpeg), the endpoint falls back to the raw recording.

## Soft-fail behavior

| Failure mode | What happens to your run |
|---|---|
| `ffmpeg` not installed in the container | Run completes normally; `video.error: "ffmpeg-not-installed"`; `record_video` ignored |
| ffmpeg crashes mid-recording | Run completes; `video.error: "empty-output"` |
| Polish (title/captions/overlays) fails to compose | `polished_path` absent; endpoint serves raw |
| Both raw and polished missing | `GET /v1/runs/{id}/video` returns 404 |

In every case the actual run data (leads, costs, summary) is preserved and accessible via `{"action": "result"}`.

## Per-tenant isolation

Recordings live at `$MANTIS_DATA_DIR/tenants/<tenant_id>/runs/<run_id>/recording_polished.<fmt>`. The download endpoint scopes the file lookup to the authenticated tenant's dir — guessing another tenant's `run_id` won't return their video.

## See also

- [Reference / HTTP API](../api.md#screencast-video-recording) — endpoint-level detail
- [Plans](plans.md) — full request body shape
