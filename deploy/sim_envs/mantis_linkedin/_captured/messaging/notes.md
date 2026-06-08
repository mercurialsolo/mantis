# /messaging/

## Layout

Two-pane "focused inbox" pattern in a single card spanning 1056px wide,
720px tall. Left pane 320px (thread list), right pane fills the rest.

## Left pane (thread list)

- Header: "Messaging" + filter dropdown + edit pencil icon.
- Search bar (pill, bg #edf3f8).
- Tabs: "Focused" (active blue underline) | "Other".
- Thread rows. Each row: avatar 56px, name 14px 600, snippet 13px grey
  (one-line truncated), timestamp 12px grey (right). Unread → bold name
  + blue dot indicator.

## Right pane (thread)

- Header: avatar 32px + name (14px 600) + headline grey caption + 3-dot menu.
- Messages list: bubbles, sender on right (blue bg #0a66c2 text white),
  recipient on left (bg #e5e5e5 text near-black).
- Each bubble: 12px radius (top-right 4px when right sender),
  padding 10px 12px, max-width 60%.
- Composer at bottom: "Write a message…" textarea + attach/gif/photo icons
  + blue "Send" button (disabled when empty).

## Interactions

- Send: POST `/messaging/<thread>/send` → bubble appears, thread bumps to top.

## Styles

- Active thread row: bg `#eef3f8`.
- Message bubble timestamps appear on hover.
