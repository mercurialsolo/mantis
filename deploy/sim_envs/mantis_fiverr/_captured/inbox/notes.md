# /inbox + /inbox/<thread> — Fiverr messaging

## /inbox — list view
Two-pane: left thread list 360px wide, right empty state placeholder
"Select a conversation".

### Thread list
- Header "Inbox" 20px / 700, padding 16px
- Search input "Search messages" 36px tall, gray bg `#fafafa`
- Each row: 56px tall, padding 12px 16px, hover bg `#fafafa`, selected
  3px green left border
  - 40px avatar
  - Right of avatar (stacked): name 14px / 600, last message snippet
    13px / `#74767e` (1 line clamp)
  - Right edge: timestamp 12px / `#95979d` + unread dot (green
    `#1dbf73`, 8px)

## /inbox/<thread> — thread detail
Same nav. Two-pane: same left list 360px, right thread 660px+.

### Right thread header
- 40px avatar + seller name + green online dot + "Online" 12px /
  `#1dbf73`. Right side: "View profile" link.

### Messages area
- Scrollable, padding 24px
- Each message bubble: max-width 480px
  - Own messages: right-aligned, bg `#dbf5e7`, text `#222325`,
    rounded 12px, padding 8px 12px
  - Other: left-aligned, bg `#f5f5f5`, text `#222325`
- Timestamp small caption 11px / `#95979d` below bubble cluster
- Avatar shown only on other-party bubble (left edge)

### Composer (bottom, sticky)
- Border-top 1px `#e4e5e7`, padding 16px, white bg
- Textarea (height 60px, autosize), placeholder "Type your message..."
- Right: attachment paperclip + send button (green, 36×36, paper-plane
  icon, disabled when empty)

## Interactions
- Send → POSTs `/inbox/<thread>/send` (form) → appends row + redirects
  back to thread
