# `/apply/<id>` — Indeed Easy Apply (3 steps)

## Step 1 — Contact

- Card width 720px, centred.
- H1 `Add your contact information`.
- Stepper above H1: `1. Contact` (active) → `2. Resume` → `3. Questions`
  → `4. Review`. Active step is brand-blue, inactive is grey.
- Fields:
  - Full name (text, required).
  - Email (email, required, prefilled when logged in).
  - Phone (tel, required).
  - City (text), State (select), Zip (text).
- `Continue` blue button bottom right; `Back` outline button left.

## Step 2 — Resume

- H1 `Add your resume`.
- List of existing resumes as radio cards (`Use resume_00001`).
- Or upload (file input + drop zone).
- `Continue`.

## Step 3 — Questions

- H1 `Answer screening questions`.
- 2-3 yes/no or short-answer questions seeded per job.
- `Continue`.

## Step 4 — Review

- H1 `Review your application`.
- Summary of all fields, with `Edit` links per section.
- `Submit application` button (brand blue).
- Submitting → POST `/apply/<jk>` → `application` row + redirect to
  `/apply/<jk>/confirm` showing success message.

## Capture status

- Synthetic spec — Indeed's real Easy Apply flow has varying step counts
  per job; we ship the canonical 3-step shape.
