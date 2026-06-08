# mantis-linkedin — in-scope spec

This sim env mirrors **linkedin.com** as a deterministic, server-rendered
replica of the professional-network surface. The mantis CUA agent will
drive it through Chrome (vision-grounded) and the oracle harness will
grade actions against the audit_log.

Live home URL: https://www.linkedin.com/

## Surface

The mirror covers four flows: **profile**, **feed**, **jobs**,
**connections**, plus **messaging** as the cross-flow surface.

## Core entities

- `users` — profile (headline, about, experience[], education[], skills[])
- `posts` — feed posts (text, optional #hashtag), authored by a user
- `comments` — comment on a post, authored by a user
- `reactions` — per-post per-user (like, celebrate, support, love, insightful, funny)
- `connections` — directed accept/pending (from_user → to_user, status, note)
- `messages` — within a `thread`, ordered, plain text
- `jobs` — title, company, location, easy_apply flag, description_md
- `job_applications` — user × job, status='submitted', payload (phone, resume_id)

## In-scope pages

- `/feed/`  — main feed (left rail profile-card + saved-items, centre feed of posts, right rail people-you-may-know + news)
- `/in/<username>/` — profile (sticky header with name+headline+actions, About, Experience, Education, Skills, Activity)
- `/mynetwork/` — connections home (manage + invitations summary cards)
- `/mynetwork/invitation-manager/` — received + sent invitations tabs
- `/messaging/` — left thread list, right thread pane + composer
- `/jobs/` — jobs home (recommended for you + recent searches)
- `/jobs/search/?keywords=` — search results (left filter, middle list, right detail pane)
- `/jobs/view/<id>/` — job detail (Easy Apply or Apply on company site)
- `/login`, `/signup` — auth pages

## In-scope interactions

- **Connect** — profile page → Connect button → 'Add a note' modal → Send → invitation row + audit
- **Post** — feed → Start a post → modal → submit → post appears top of feed + audit row
- **React** — like a post → reactions row + count bump
- **Comment** — leave a comment on a post → comments row + count bump
- **Easy Apply** — job detail → Easy Apply modal (contact → resume → screening → review → submit) → application row + audit
- **Message** — open a thread → type → send → message row + thread bump

## Out of scope (explicitly NOT mirrored)

- Real brand assets / photography (placeholder SVGs only)
- Advertising, analytics, tracking scripts
- Third-party widgets (Intercom, Hotjar)
- OAuth / social login (username + password only)
- Payment processing (Premium upsell stripped)
- Real-time websockets / SSE (polling or static)
- A/B variant pages (pick the canonical anonymous-visitor variant)
- Groups, Pages, Events, Premium, Sales Nav, Learning, Newsletters, Live Video

## Done bar

- Pixel diff per section vs the live site captured in `_captured/<slug>/`: ≤2px structural deltas, <0.5% perceptual diff (NORTH STAR; first-pass target is "close (matches structurally; minor pixel diff)" per FIDELITY.md).
- Every in-scope interaction replays through HTTP and writes an audit_log row matching the oracle contract.
- `scripts/smoke.py` passes — boot, every in-scope page returns 200, each oracle dispatches and returns the JSON shape, the happy path for each task drives the oracle to `passed=true`.
- FIDELITY.md lists every in-scope page with ≥5 element rows and an honest status per row.
