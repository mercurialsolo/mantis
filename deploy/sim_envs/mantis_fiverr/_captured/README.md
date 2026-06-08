# _captured/ — fiverr.com surface corpus

Spec source: **training-data recollection** (capture brief
`deploy/sim_envs/_capture_brief.md` was not present at worktree
start; Chrome MCP / WebFetch capture against fiverr.com is gated by
their bot wall + heavy JS rehydrate, so a live DOM grab in one turn
was not feasible).

Each subfolder mirrors a logical page; `notes.md` carries the layout
+ palette + interaction spec. `screenshot.png` is intentionally
omitted on this first pass (no live grab) — flagged ⏳ in
FIDELITY.md. A follow-up agent should drive Chrome MCP against the
live site and overwrite the per-slug notes with measured values.

## Canonical Fiverr palette + typography (training-data recollection)

| Token                  | Value          | Notes                              |
| ---------------------- | -------------- | ---------------------------------- |
| `--fv-green-primary`   | `#1dbf73`      | Primary CTA, logo, hover accents   |
| `--fv-green-hover`     | `#19a463`      | Hover on primary                   |
| `--fv-green-dark`      | `#16805a`      | Active state                       |
| `--fv-text-primary`    | `#222325`      | Body text                          |
| `--fv-text-secondary`  | `#74767e`      | Meta, captions                     |
| `--fv-text-tertiary`   | `#95979d`      | Disabled, separators               |
| `--fv-bg-white`        | `#ffffff`      | Card/page bg                       |
| `--fv-bg-gray`         | `#fafafa`      | Section bg, search input fill      |
| `--fv-border-gray`     | `#e4e5e7`      | Cards, hr, input border            |
| `--fv-yellow-star`     | `#ffb33e`      | Rating stars                       |
| `--fv-orange-pro`      | `#ff7640`      | Fiverr Pro badge accent            |
| `--fv-purple-business` | `#5b3eff`      | Fiverr Business accent             |

Typography: **Macan** in production; we substitute `system-ui` /
Helvetica Neue / Arial. Sizes:

- H1 hero: 48px / 700 / 1.1 line-height
- H2 section: 32px / 700
- H3 card title: 18px / 700
- Body: 14px / 400
- Caption: 12px / 400 / `#74767e`
