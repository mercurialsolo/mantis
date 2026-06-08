# `/employers/jobs/new` — create posting

## Layout

- H1 `Post a job`.
- Form (single column 720px):
  - Job title (text, required).
  - Company (preselected to employer's company).
  - Location (text, required).
  - Salary low + Salary high (number inputs side by side).
  - Remote (checkbox).
  - Job type (select: Full-time, Part-time, Contract, Internship).
  - Description (textarea, required).
- `Publish` blue primary button + `Save as draft` outline.

## Interaction

- POST `/employers/jobs/new` → jobs row + audit row + redirect to
  `/employers/jobs/<posting_id>`.
