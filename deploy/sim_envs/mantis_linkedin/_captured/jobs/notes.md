# /jobs/ — jobs home

## Layout

Two-column. Left rail 225px, centre 860px.

## Left rail

- "Job collections" header.
- Links: My jobs, Preferences, Post a free job, Interview prep,
  Salary explorer, Job seeker guidance.

## Centre column

### Search bar card

- "Search by title, skill or company" input (pill, white bg).
- "City, state or zip" input adjacent.
- "Search" filled-blue button.

### Top job picks for you card

- H2 "Top job picks for you".
- Caption "Based on your profile and search history".
- Horizontal carousel of 6 job cards. Each card 240px wide:
  company logo 48px + title (14px 600 blue) + company (12px) + location +
  promoted/Easy Apply tag.

### Recommended for you card

- H2 "Recommended for you".
- Vertical list of 5 job rows. Each row: logo 48px + title + company +
  location + caption "N hours ago • N applicants".

### Recent job searches card

- H2 "Recent job searches".
- List of saved searches with edit/delete icons.

## Interactions

- Click a job card → navigates to `/jobs/view/<id>/`.
- Click "Search" → `/jobs/search/?keywords=...&location=...`.
