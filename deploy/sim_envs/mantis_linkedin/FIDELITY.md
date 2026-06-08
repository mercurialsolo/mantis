# mantis-linkedin fidelity tracker

**Last updated:** 2026-06-08

Per-page match matrix vs the captured spec in `_captured/<slug>/notes.md`.
Status legend (mantis_boattrader convention):

- **exact** — pixel-parity (≤2px structural delta, <0.5% perceptual diff).
- **close** — matches structurally; minor pixel diff in spacing/weights/radius.
- **partial** — visible delta but recognisable as the same element.
- **missing** — not yet implemented.
- **not-matched** — implemented differently / intentionally simplified.

## Method note

The corpus in `_captured/` was authored offline from training-data
recollection because linkedin.com requires authentication and challenges
automated browsers on most surfaces. A Chrome-MCP–driven capture pass
should be run later to validate the offline spec; FIDELITY rows below
are confident but unverified vs the live site. Any disagreement found
during that pass should land here as new rows + updated status.

## Global brand / shell

| Element                     | Real spec                                                                                | Mine                                  | Status |
| --------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------- | ------ |
| Page background             | `#f3f2ef`                                                                                | `#f3f2ef`                             | exact  |
| Body font stack             | `-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial`                              | same                                  | exact  |
| Body type                   | 14px / 1.4286 / rgba(0,0,0,0.9)                                                          | same                                  | exact  |
| Primary blue                | `#0a66c2` / hover `#004182`                                                              | same                                  | exact  |
| Primary button              | 9999px pill, 600 weight, 6/16 padding                                                    | same                                  | exact  |
| Card surface                | `#fff`, 8px radius, `0 0 0 1px rgb(0 0 0 / 8%), 0 2px 3px rgb(0 0 0 / 8%)`               | same                                  | exact  |
| Top nav height              | 52px fixed, content max-width 1128px                                                     | same                                  | exact  |
| Top nav `in` mark           | Blue square 32×32, 4px radius, 700 weight                                                | same                                  | exact  |
| Top nav search              | Pill 280px, `#edf3f8` bg, search-icon left, 14px label                                   | same                                  | close  |
| Top nav icon stack          | Home / My Network / Jobs / Messaging / Notifications / Me                                | same order + glyph + caption          | close  |

## /feed/

| Element                       | Real spec                                                                | Mine                                              | Status   |
| ----------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- | -------- |
| Layout grid                   | 225 / 555 / 300 with 24px gaps under 1128px max-width                    | same                                              | exact    |
| Left rail — profile card      | Banner 60px gradient + 72px avatar overlap + name/headline + Connections | banner 56px, otherwise matches                    | close    |
| Left rail — My items mini     | Bookmark glyph + label                                                   | bookmark + label                                  | close    |
| Centre — Start a post         | Avatar 48px + pill input + 4-up action row (Photo/Video/Event/Article)   | same                                              | close    |
| Centre — post card author row | 48px avatar + name + headline + time + globe icon + ⋯ menu               | same                                              | close    |
| Centre — post counters row    | reaction blob + count + "N comments"                                     | same                                              | close    |
| Centre — post action bar      | Like / Comment / Repost / Send 4-up                                      | same                                              | close    |
| Right rail — Add to your feed | Header + 3 suggested follows w/ Connect pill                             | same shape, "Follow" pill                         | close    |
| Right rail — News card        | Stack of 5 stories                                                       | same                                              | close    |
| Start-a-post modal            | Avatar/name header, textarea, blue Post button (disabled when empty)     | same                                              | close    |

## /in/<handle>/

| Element                | Real spec                                                          | Mine                                          | Status   |
| ---------------------- | ------------------------------------------------------------------ | --------------------------------------------- | -------- |
| Hero banner            | 200px gradient strip                                               | same                                          | exact    |
| Hero avatar            | 160px circle, 4px white border, overlapping banner                 | same                                          | exact    |
| Hero name + headline   | H1 24/600 + 14/400 grey                                            | same                                          | exact    |
| Hero actions           | Connect (primary), Message (secondary), More                       | same; Open to/Add section variants for self   | close    |
| About card             | H2 + body                                                          | same                                          | close    |
| Experience rows        | 48px company logo + title + company + dates + location + body      | same                                          | close    |
| Education rows         | School logo + school + degree + dates                              | same                                          | close    |
| Skills section         | Skill name + "Endorsed by N connections"                           | same                                          | close    |
| Activity card          | Header + recent posts                                              | header + last 5 posts; no tabs yet            | partial  |
| Connect modal          | Title "Add a note", 300-char textarea, Cancel + Send blue          | same                                          | close    |

## /mynetwork/

| Element                       | Real spec                                  | Mine                          | Status |
| ----------------------------- | ------------------------------------------ | ----------------------------- | ------ |
| Manage-my-network rail        | Connections / Contacts / Following / etc.  | same items + counts           | close  |
| Invitations card              | Up to 3 with Accept/Ignore                 | same                          | close  |
| People you may know grid      | Mini cards w/ banner + avatar + Connect    | same                          | close  |
| Mini card Connect button      | Outlined pill                              | same                          | close  |

## /mynetwork/invitation-manager/

| Element            | Real spec                                                | Mine                            | Status |
| ------------------ | -------------------------------------------------------- | ------------------------------- | ------ |
| Tabs               | Received / Sent / I don't know                           | same                            | exact  |
| Meta row           | "N pending invitations" + sort dropdown                  | same                            | close  |
| Row layout         | Avatar 56 + name + headline + actions on right           | same                            | close  |
| Sent tab actions   | Withdraw (outlined)                                      | same                            | close  |
| Received actions   | Ignore (outlined) + Accept (primary)                     | same                            | close  |

## /messaging/

| Element              | Real spec                                                 | Mine                                | Status   |
| -------------------- | --------------------------------------------------------- | ----------------------------------- | -------- |
| Two-pane card        | 320 / rest, total 1056×720                                | same                                | close    |
| Thread list header   | Title + filter + compose pencil                           | same                                | close    |
| Thread search        | Pill, `#edf3f8`                                           | same                                | exact    |
| Focused/Other tabs   | Blue underline on active                                  | same                                | exact    |
| Thread row           | Avatar 56 + name + snippet + timestamp                    | uses 48px avatar; otherwise matches | close    |
| Active thread row    | Bg `#eef3f8`                                              | same                                | exact    |
| Pane header          | Avatar + name + headline + ⋯                              | same                                | close    |
| Message bubbles      | Self right blue, peer left grey, 12px radius corner-clip  | same                                | close    |
| Composer             | Textarea + tool row + blue Send                           | same                                | close    |

## /jobs/

| Element             | Real spec                                                  | Mine                             | Status |
| ------------------- | ---------------------------------------------------------- | -------------------------------- | ------ |
| Rail "Job collections" | Links list w/ glyph + label                              | same                             | close  |
| Search card         | Title + Location + Search button                           | same                             | close  |
| Top picks carousel  | 6 mini cards 240px wide                                    | grid auto-fill min 240px         | close  |
| Recommended list    | 5 rows w/ logo + title + meta                              | same                             | close  |

## /jobs/search/

| Element             | Real spec                                                  | Mine                             | Status   |
| ------------------- | ---------------------------------------------------------- | -------------------------------- | -------- |
| Filter chip row     | Sticky chips                                               | same; 4px shorter                | close    |
| Left list           | 420px, selected row 4px left blue border                   | same                             | close    |
| Right detail        | Sticky header card + tabs + body                           | header + tabs + md-body          | close    |
| Easy Apply pill     | Filled blue header pill on detail                          | same                             | close    |

## /jobs/view/<id>/

| Element                  | Real spec                                                       | Mine                                 | Status   |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------ | -------- |
| Centre header card       | Logo 64 + title 24/600 + company + meta + Easy Apply primary    | same                                 | close    |
| About card               | Markdown rendered body                                          | pre.md-body (preserves md formatting)| close    |
| Company rail card        | 96px logo + name + tagline + followers + Follow                 | same                                 | close    |
| Easy Apply modal width   | 520px fixed                                                     | 552px max via `<dialog>`              | close    |
| Easy Apply progress      | 4 dots, active blue                                             | same                                 | close    |
| Easy Apply steps         | Contact → Resume → Screening → Review                           | same; vanilla JS step machine        | close    |
| Submit success           | "Application submitted!" toast                                  | redirect → ?applied=1; no toast yet  | partial  |

## /login

| Element             | Real spec                                                                          | Mine                       | Status |
| ------------------- | ---------------------------------------------------------------------------------- | -------------------------- | ------ |
| Card width          | 400px, centred                                                                     | same                       | exact  |
| Brand mark above    | 32px square logo                                                                   | same                       | exact  |
| Inputs              | 12px padding, 4px radius, focus = 2px blue outline                                 | same                       | close  |
| Sign-in button      | Filled blue pill, 48px tall, 16px 600                                              | same                       | close  |
| OAuth buttons       | Continue with Google / Apple (outlined)                                            | omitted (out of scope)     | not-matched |
| Footer "Join now"   | Centred 14px link                                                                  | same                       | exact  |

## Interaction parity

| Interaction         | Real behaviour                                                | Mine                                                     | Status |
| ------------------- | ------------------------------------------------------------- | -------------------------------------------------------- | ------ |
| Start a post        | Modal → submit → post appears top of feed                     | same; audit_log `post_created`                           | close  |
| Like a post         | Toggle reaction + count bump                                  | same; audit_log `reaction_added` / `reaction_removed`    | close  |
| Comment on a post   | In-place composer reveal → submit                             | same; audit_log `comment_added`                          | close  |
| Connect             | Profile → modal → Send → invitation row                       | same; audit_log `connection_requested`                   | close  |
| Easy Apply          | 4-step modal → submit → application row                       | same; audit_log `job_application_submitted`              | close  |
| Send message        | Composer → bubble appears + thread bumps                      | same; audit_log `message_sent`                           | close  |
| Accept invitation   | Row removed + Connections count bump                          | same; audit_log `connection_accepted`                    | close  |

## Iteration log

- **2026-06-08 v1** — Phases 0–6 first pass. All in-scope pages render
  200 OK; oracle dispatch + happy paths for t01/t02/t03 pass via
  `scripts/smoke.py`. Brand palette + typography exact-matched from
  the offline spec. Visual fidelity at "close" majority; chasing
  "exact" deferred until a live Chrome-MCP capture pass can validate
  per-pixel deltas.

## Open gaps (carry to next pass)

- No live screenshots captured — `_captured/<slug>/screenshot.png` absent.
  Run a Chrome-MCP capture pass against a real LinkedIn session and update
  rows that disagree.
- Activity tabs on `/in/<handle>/` (Posts / Comments / Videos / Images) — only
  Posts implemented as a flat list. Tab UI is `partial`.
- Easy Apply submit success uses a `?applied=1` URL flag instead of an in-page
  toast — `partial`.
- OAuth login buttons intentionally omitted; flagged `not-matched`.
- Notifications icon in topnav is a stub link — no `/notifications` page yet.
