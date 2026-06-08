# /jobs/search/?keywords=...

## Layout

Three-pane. Top filter bar (sticky). Left list 420px, right detail 700px.

## Top filter chip row

- Sticky under the topnav (52+44=96px from top).
- Chips: "Date posted ▾", "Experience level ▾", "Company ▾",
  "Job type ▾", "Remote ▾", "Easy Apply" (toggle), "All filters".

## Left list (420px)

- Header: "<N> results for "<keyword>"" + sort dropdown (Most relevant / Most recent).
- List of job result cards. Each row 88px tall: company logo 56px left,
  title (16px 600 blue) + company (14px) + location (12px grey) +
  "Easy Apply" tag (when applicable) + caption "N applicants • N hours ago".
- Selected row: bg `#f3f2ef`, 4px left border `#0a66c2`.

## Right detail (700px)

- Sticky header card: logo + title (24px 600) + company + location +
  applicants + "Easy Apply" filled-blue pill (or "Apply" outlined-blue for external).
- Tabs: About the job (default), Hiring team, Related jobs.
- About body: description_md rendered as paragraphs + bulleted requirements.
- Bottom: "How your profile matches" mini card.

## Interactions

- Click a job row → updates right pane URL `?id=<job_id>`. No full nav.
- Easy Apply opens multi-step modal (see `_captured/jobs_view/`).
