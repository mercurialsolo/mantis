# `/dashboard` — Role-sensitive

## Candidate view

- H1: `My applications`
- Table: Role / Company / Status / Applied at / Actions
- Each row's Status is one of: submitted / under_review / interview /
  rejected / hired.

## Client view

- H1: `Review queue`
- Two side-by-side panels:
    - Left: pending applications for postings I own. Each row has
      `Add to shortlist` and `Decline` buttons.
    - Right: shortlist count + the most recent 5 shortlisted
      candidates.
- Decline button opens an inline form prompting `Reason` (required).

## Mirror priorities

- Role sniffed from session.users.role.
- Add-to-shortlist button toggles to `Added` inline (no nav).
- Decline reason required before submit enables.
