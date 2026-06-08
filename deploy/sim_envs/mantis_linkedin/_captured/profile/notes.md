# /in/<username>/ — profile

## Layout

Two-column under top nav. Centre column 760px, right rail 300px (omitted on /in/).
Full-width banner card up top.

## Hero card

- Banner image strip 200px tall (gradient placeholder).
- Avatar 160px circle, bottom-left, overlapping banner.
- Name H1 24px / 600.
- Headline 14px / 400 grey.
- Location row (14px grey, with country dot bullet).
- "N connections • Contact info" inline link (12px blue).
- Actions row: "Open to", "Add profile section", "Enhance profile", "More".
  When viewing a different user: "Connect" (primary), "Message" (secondary), "More".

## About section card

- H2 "About" (20px 600).
- Pencil edit icon top-right (own profile only).
- Body 14px, expandable "...see more" after 3 lines.

## Experience section card

- H2 "Experience" + edit icon.
- List of roles. Each row: company logo 48px square + title (14px 600),
  company (14px 400), dates (12px grey), location (12px grey), description.
- Multiple roles at one company nested under company header.

## Education section card

- H2 "Education" + edit icon.
- Each row: school logo 48px square + school (14px 600), degree (14px),
  dates (12px grey).

## Skills section card

- H2 "Skills" + edit icon.
- List of skills with "Endorsed by N connections" caption.

## Activity card

- H2 "Activity" + "N followers" + "Create a post" button (right).
- Tabs: Posts, Comments, Videos, Images.
- 3 recent post cards.

## Interactions

- Connect (primary CTA on others' profiles): opens modal
  `#connect-modal` with copy "How do you know <Name>?", a 300-char textarea
  (placeholder "Add a note"), Cancel + Send buttons.
- Message: navigates to `/messaging/?to=<username>`.

## Styles

- Hero card same shadow/radius as feed cards.
- Section headers 20px 600 with 16px margin-bottom.
