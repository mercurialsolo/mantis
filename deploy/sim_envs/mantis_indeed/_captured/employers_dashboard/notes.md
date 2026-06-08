# `/employers/dashboard` — employer dashboard

## Layout

- Sticky top nav (employer variant): brand wordmark + `Post a job` /
  `Dashboard` / `Sign out`.
- H1 `Employer dashboard`.
- KPI strip (3 cards): `Active postings`, `New applicants this week`,
  `Total views`.
- Posting list table:
  - Columns: `Title`, `Location`, `Applicants` (with badge breakdown
    New/Reviewed/Rejected/Hired), `Posted`, `Status`.
  - Row click → `/employers/jobs/<posting_id>`.

## Applicant detail (`/employers/jobs/<posting_id>`)

- H1 `{Job Title}`.
- Applicant table:
  - Columns: `Name`, `Resume`, `Status`, `Applied`, Actions.
  - Status as a select dropdown: `New`, `Reviewed`, `Rejected`, `Hired`.
  - Saving the dropdown → POST `/employers/applications/<id>/status` →
    `applications.status` + `reviewed_at` + audit row.

## Interaction

- Filter applicants by status via top-of-table chip strip.
- Click `New` applicants -> filter -> select `Reviewed` from row select
  → status persists.
