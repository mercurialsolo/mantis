# /feed/ — main feed

## Layout

Three columns under a fixed top nav. Page bg `#f3f2ef`. Content max-width 1128px,
horizontally centred. Top nav 52px tall.

```
┌── top nav (logo + search + Home/My Network/Jobs/Messaging/Notifications/Me) ─┐
│                                                                              │
│  [left rail 225]   [centre feed 555]   [right rail 300]                       │
│                                                                              │
│  - profile card    - "Start a post" card   - Add to your feed                 │
│  - groups item     - posts (scroll)        - LinkedIn News                    │
│  - events item                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Top nav (52px, white bg, bottom border 1px #e0dfdc)

Left: LinkedIn logo (`in` mark blue square). Search box: pill shape, bg `#edf3f8`,
placeholder "Search", width ~280px, left-padding 40px (search icon).

Right (icon-stack, vertical icon + caption label):
- Home (filled when active, underline bar 2px below icon)
- My Network
- Jobs
- Messaging
- Notifications
- Me (avatar circle + caret)
- Divider
- For Business
- Try Premium for free (link, gold colour)

## Left rail (profile card)

White card, 8px radius, drop shadow.
- Banner strip (60px tall, brand-grad blue).
- Avatar circle 72px, centred horizontally, overlapping banner by 36px.
- Name (16px bold, centred).
- Headline (12px regular, color rgba 0,0,0,0.6).
- Below divider: "Connections" label + count (right-aligned), then
  "Grow your network" caption.
- Second card: "My items" with bookmark icon.

## Centre feed

Top card: "Start a post"
- Avatar 48px left, single-line input bar (placeholder "Start a post"), pill button.
- Action row: Photo (icon green), Video (icon teal), Event (icon orange),
  Write article (icon orange-red).

Each post card:
- Author chip: avatar 48px, name (14px 600), headline (12px regular grey),
  timestamp + " • " + globe icon.
- Body text (14px regular, max-3 lines truncated with "…see more" link).
- Optional media slot (omitted in mirror).
- Counters row: like-icon + count, "N comments • M reposts".
- Action bar: Like, Comment, Repost, Send (each icon + label, 4-up grid).

## Right rail

Card 1 — "Add to your feed": header, then 3 suggested follows
(avatar + name + headline + "+ Follow" pill).

Card 2 — "LinkedIn News" with stacked stories
(title 14px 600, "N hours ago • N readers" caption).

## Interactions

- "Start a post" opens modal `#start-post-modal`. Modal has avatar+name header,
  textarea (placeholder "What do you want to talk about?"), bottom action bar,
  blue "Post" button (disabled until text non-empty).
- Like button: click toggles. Heart fills, count bumps.
- Comment button: scrolls/reveals comment composer under post.

## Computed-style highlights

- Body: `font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif`;
  font-size 14px; color rgba(0,0,0,0.9); line-height 1.42857.
- Cards: `background: #fff; border-radius: 8px;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.08), 0 2px 3px rgba(0,0,0,0.08);`
- Primary action button: bg #0a66c2; color #fff; border-radius: 9999px;
  padding 6px 16px; font-weight 600; hover bg #004182.
- Secondary button: 1px solid #0a66c2; color #0a66c2; transparent bg.
