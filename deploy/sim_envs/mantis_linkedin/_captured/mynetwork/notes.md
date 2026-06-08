# /mynetwork/ — connections home

## Layout

Two-column. Left rail 225px ("Manage my network" links), centre 860px.

## Manage my network rail

- Header "Manage my network" (16px 600).
- Rows: Connections (count), Contacts, Following & followers, Groups, Events, Pages.
  Each row icon left, label, count right (12px grey).

## Centre column

### Invitations card

- H2 "Invitations" + "Show all (N)" link.
- 0..3 invitation rows. Each row: avatar 56px, name + headline, two pill buttons
  ("Ignore" outlined, "Accept" filled blue).

### People you may know

- H2 "People you may know based on your profile".
- Horizontal scroll-list of mini cards. Each card 200px wide:
  avatar 72px centred, name, headline (2-line truncated),
  "Connect" outlined-pill, bottom-spanning.

## Interactions

- Accept: green flash on row, removes after 200ms, count bumps Connections.
- Ignore: row removed.
- Connect (mini card): adds outgoing invitation, button → "Pending".

## Styles

- Cards same as feed.
- Mini card hover: lift 2px, border darker.
