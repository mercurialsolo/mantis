# /orders + /orders/<id> — Fiverr orders

## /orders — buyer order list

Single-column, max-width 1108px, padding 0 80px.

### Header row
- H1 "Manage Orders" 28px / 700
- Right: "Active (N)" "Completed (N)" "Cancelled (N)" tabs

### Filter chips
Chips row: All · Active · Delivered · Completed · Cancelled (pill,
border 1px `#e4e5e7`, selected: green `#1dbf73` bg / white text)

### Table
Columns: BUYER (if seller view) / SELLER (if buyer view) | GIG |
DUE ON | TOTAL | STATUS | ACTIONS
- Header row: 12px / 600 / `#74767e`, uppercase
- Row: white, hover `#fafafa`, 64px tall, gig thumb 48×48 + title
  truncate, status pill (Active = blue, Delivered = orange, Completed
  = green, Cancelled = gray)

## /orders/<id> — order detail

Three sections stacked, max-width 980px:

### Order header card
- Order # 22px / 700
- Status pill + due-on info + gig title link
- "Total: US$X" right
- Action row: secondary buttons "Message seller", "Resolve order"

### Deliverables panel
- Empty state "Awaiting delivery..." if status=active
- If status=delivered: list of file rows (name, size, download icon)
- If status=completed: same + "Mark as completed" already disabled

### Review CTA (only when status=completed AND no review yet)
- H3 "Leave a review" 16px / 700
- Star input row: 5 outline stars, clicking fills 1..5
- Textarea "Share more about your experience..."
- Green submit button "Publish review"

## Interactions
- "Publish review" → POST `/orders/<id>/review` with stars+text →
  inserts review row + recomputes gig avg rating + audit_log row +
  redirects
