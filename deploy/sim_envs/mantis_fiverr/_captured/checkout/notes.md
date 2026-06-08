# /checkout/<gig-id>?tier=… — Fiverr checkout

## Layout

Two-column, max-width 980px, padding 0 80px. Left 600px (order
summary + form), right 320px sticky summary card.

### Header
Compact: Fiverr logo only, no nav. Right side "Secure checkout 🔒"
14px / `#74767e`.

### Order summary block (left)
- H1 "Checkout" 28px / 700
- "You're ordering:" 14px / `#74767e`
- Gig row: 56×56 thumb + title + seller + price 18px / 700 right-aligned
- Selected tier line: "Package: Basic — Delivery in 3 days"
- "Order details" textarea, label 14px / 600, placeholder "Briefly
  describe your project (optional)"

### Payment block
- H2 "Payment method" 20px / 700
- Radio cards: `Credit / Debit Card` (selected), `PayPal`, `Bank
  transfer`
- Selected variant exposes card-number / expiry / CVC fields — placeholder
  values (no validation, since payment is mocked)

### Footer CTA bar
Sticky bottom on left col: green button "Confirm & Pay (US$X)" 48px
tall, full-width.

### Right summary card
- H3 "Summary" 16px / 700
- Subtotal row, Service fee row, Total row (16px / 700)
- Small print "By clicking Confirm & Pay you agree to our Terms…"
  12px / `#95979d`

## Interactions captured
- "Confirm & Pay" POSTs to `/checkout/<gig_id>` → creates order row +
  audit_log row, redirects 303 to `/orders/<order_id>`
