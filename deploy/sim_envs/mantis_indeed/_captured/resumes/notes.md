# `/resumes` — resume manager

## Layout

- Sticky top nav.
- H1 `Your resumes`.
- List of resume cards, each with:
  - Title input (editable inline).
  - `Last updated 2 days ago` microcopy.
  - `Edit` and `Delete` outline buttons.
- `+ Add new resume` button (blue primary) below the list.
- New resume form (modal-like inline panel): title, summary textarea,
  skills (comma list), experience (textarea), `Save`.

## Interaction

- POST `/resumes/new` → resume row + audit row + back to list.
- POST `/resumes/<id>/edit` → patch row + audit.
- POST `/resumes/<id>/delete` → soft delete + audit.
