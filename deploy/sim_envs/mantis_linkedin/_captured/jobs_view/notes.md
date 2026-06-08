# /jobs/view/<id>/

## Layout

Two-column. Centre 720px (job content card), right rail 300px (company info).

## Centre column

Single card stack:

### Header card

- Company logo 64px + title H1 (24px 600) + company link (14px blue) +
  location + "Posted N days ago • N applicants".
- Action row: "Easy Apply" (filled blue, 36px tall) or "Apply" outlined +
  "Save" bookmark icon-button + "Share" link.

### About the job card

- "About the job" header (16px 600).
- Description body (markdown rendered).
- "Show more" link if truncated.

### How your profile matches card

- Skills checklist, applicants graph etc. (mirror omits).

## Right rail

- Company logo 96px + name + tagline + N followers + "Follow" button.

## Easy Apply modal (multi-step)

Trigger: click "Easy Apply" → opens dialog `#easy-apply-modal`.

Steps (top progress bar, 4 dots):
1. **Contact info** — name (read-only), email (read-only), phone (input req'd),
   country dropdown. "Next" button (disabled until phone non-empty).
2. **Resume** — radio "Use my profile" (default) or "Upload resume". "Next".
3. **Screening questions** — 1-2 text inputs (e.g. "Years of experience?"). "Next".
4. **Review** — summary list of all collected fields. "Submit application" filled blue.

POST `/jobs/<job_id>/apply` with phone + answers → writes
applications row + audit, modal flips to "Application submitted!" success.

## Styles

- Modal: width 520px, max-height 720px, white bg, 8px radius, backdrop
  rgba(0,0,0,0.6).
- Progress bar: 4 dots, active blue, inactive grey.
