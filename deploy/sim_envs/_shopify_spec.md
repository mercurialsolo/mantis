# mantis_shopify — Shopify Partners mirror spec

Captured 2026-06-08 from a live Shopify Partners dashboard via Chrome
MCP. The aim is a functional, high-fidelity mirror of the Partners
back office IA — usable as a CUA training sim env with oracles. The
**App distribution / `/apps`** sub-section is intentionally excluded
per scope.

Path prefix in the mirror: all routes hang off `/`. Partners uses
`/<partner_id>/<section>` in production; we drop the partner-id
segment since the env is single-tenant.

---

## Global chrome

- **Topbar** (`<header class="sp-topbar">`)
  - Brand mark: green leaf shopping-bag glyph + "shopify partners"
    in italic Shopify Sans (use any web-safe italic serif fallback;
    `font-family: "Source Serif Pro", Georgia, serif; font-style:italic`).
  - Center: **Search input** with magnifier glyph, full-width 720px max.
  - Right cluster: **Notifications bell** with red badge (number),
    **avatar chip** with initials + name + subtitle (display name on
    one line, email/business name on a second line).
- **Left sidebar** (`<aside class="sp-sidebar">`, 232px wide)
  - **Group 1 (main, no header)**
    - Home → `/`
    - Stores → `/stores`
    - Sales → `/sales` (sub-nav: Leads, Referrals)
    - Catalogs → `/catalogs` (we DO include this — sidebar showed it)
    - Themes → `/themes`
    - Partner Directory → `/partner_directory` (sub-nav: Profile)
    - Shopify POS → `/pos`
  - **Group 2 — "Resources"** (group label small all-caps grey)
    - Partner docs → `/docs/partner`
    - Product docs → `/docs/product`
    - Support → `/support`
  - **Group 3 — "Admin"**
    - Payouts → `/payouts`
    - Team → `/team`
    - Settings → `/settings`
  - Each row: 20-px glyph + label; **active** row gets a 4-px green
    bar on the left edge + green text + light-grey background tint.
- **Page surface**: light grey body `#f6f6f7`. Content cards: white,
  `border-radius: 8px`, `box-shadow: 0 0 0 1px #e1e3e5, 0 1px 0
  rgba(0,0,0,0.05)`.
- **Primary CTA**: filled green `#008060` button, `border-radius:
  6px`.
- **Secondary button**: white with `#c9cccf` border.
- **Topbar banner row** (only on some pages): yellow `#ffeb78` strip
  with "Update emergency contact information" text + dismiss × +
  "Add contact information" inline button.

Colour palette captured:
- Sidebar bg: `#f6f6f7` (same as body).
- Active row tint: `#ebebee`.
- Active accent (left bar + text): `#008060`.
- Sidebar text default: `#202223`.
- Sidebar group label: `#6d7175`.
- Topbar bg: `#fff`.
- Topbar brand strip border-bottom: `1px solid #e1e3e5`.
- Page H1: `#202223` 24px regular.
- Subdued text: `#6d7175`.
- Link blue: `#2c6ecb`.
- Yellow banner: `#ffeb78` strip `#b54708` accent rule top.

---

## Sub-sections

### 1. Home — `/`
- **H1**: "Home"
- **Stores section**
  - White card: header row `<h2>Stores</h2>` + right-aligned green
    **Add store** primary CTA.
  - Then a stores list (top 5) with the same row layout as `/stores`
    (name bold, type pill, action dropdown + Log in link).
  - Footer link: "View all stores →"
- **Business overview** (3 columns)
  - Card 1: "Pending payout" + USD amount (e.g. `$2,151.43 USD`) +
    "View payouts" link.
  - Card 2: "Active store referrals" + number + "Submit a lead" link.
  - Card 3: "Lifetime leads submitted" + number + "View leads" link.
- **Right rail — Changelog** (`<aside class="sp-changelog">`)
  - h2 "Changelog" + small "View all" link.
  - List of 6 dated items. Each item: small grey date (`Jun 5, 2026`)
    + bold title + small grey category pill ("Admin", "Payments",
    "POS", "Analytics", "Catalogs").
  - Use neutral, generated text — do NOT mirror real changelog copy
    verbatim. Example: "Improved catalog publishing UX".

### 2. Stores — `/stores`
- **H1**: "Stores"; right-side **Add store ▾** primary, **More
  actions ▾** secondary.
- **Tabs** (full-width underline tabs): All • Client transfer •
  Collaborator access • Archived • Inactive (`/stores?tab=...`).
- Below tabs: **Search input** ("Filter stores"), **Store status**
  dropdown right-aligned.
- "Showing N stores" caption + right-aligned "Sort by Last login"
  dropdown.
- **List rows** (10 per page): each row 3 cols
  - Col 1: store name bold + subtle URL underneath
    (`mm-jop5.myshopify.com`).
  - Col 2: type pill in grey: "Client transfer" / "Collaborator
    access" / "Active development store".
  - Col 3 (right): **Actions ▾** dropdown + **Log in** link (blue,
    underline-on-hover).
- Empty-state and pagination footer.

### 3. Sales — `/sales` (sub-nav: Leads / Referrals)
- **H1**: "Leads"
- Subtitle: "Earn revenue by referring merchants to Shopify."
- **Submit a lead** card with three rows; each row: icon + product
  name (`Plus`, `Shopify POS`, `Shopify Plus B2B`) + 2-line desc +
  right-aligned outline button ("Submit a Plus lead", etc.). Buttons
  POST to `/sales/leads/new?product=plus|pos|plus_b2b`.
- **Submitted leads** card: tabbed table or empty-state
  illustration with "Earn monthly referring merchants" caption + CTA
  "Submit a lead".
- Sub-nav row above page (between sidebar and main): two tabs
  (`Leads` underlined, `Referrals` outline) — clicking "Referrals"
  goes to `/sales/referrals` which shows a table of `Referral ID /
  Merchant / Earned / Status`.

### 4. Catalogs — `/catalogs`
- **H1**: "Catalogs"
- Empty marketing-card layout: "Publish products to merchant stores
  via Catalogs API" subtitle + primary CTA "Create catalog" + two
  info cards ("How catalogs work", "API reference").

### 5. Themes — `/themes`
- **H1**: "Build themes for the Shopify Theme Store"
- Subtitle paragraph + right-aligned green CTA "Submit a theme".
- Card 1: **Build and sell your theme on the leading commerce
  platform** with 3 bullet list + illustration placeholder.
- Card 2: **Move fast with smart tooling** with 3 external links
  ("Explore developer tooling ↗", "Download Shopify CLI ↗", "Get
  started with our Skeleton theme ↗").
- Card 3: **Submit your theme for review** with a step list + CTA.

### 6. Partner Directory — `/partner_directory`
- **H1**: "Partner Directory"
- Right header: "👁 View" link + **Edit profile** primary green.
- Card "Eligible directory reviews" with paragraph + "Learn more
  about review eligibility requirements" inline link.
- **Reviews table**: cols Business Name | Plan | Review Status.
  Plan values: Plus / Grow / Basic / Partner Test / Custom. Status:
  green pill "Available". 10 rows.
- **Sub-route**: `/partner_directory/profile` — public-facing
  profile editor (form fields: Studio name, Bio, Services offered,
  Featured projects, Languages, Locations served, Hourly rate).

### 7. Shopify POS — `/pos`
- **No H1** (the title is a small caption "Shopify Point of Sale").
- **Hero**: H2 "Built for merchants selling in person and online" +
  paragraph + green CTA **Submit POS Pro referral** + outline link
  "Learn how to sell Shopify POS Pro" + "View your referrals via
  your Leads dashboard" small caption.
- Hero right: image card (green block) with "How to Earn with
  Shopify POS".
- **Three info cards** in 2-col grid:
  - "Shopify POS Marketing Toolkit" — outline CTA "Access Marketing
    Toolkit"
  - "Shopify POS Verified Skills" — outline CTA "Learn more now"
- **Learning Path** card: "Solution-Based Selling with Shopify POS"
  + outline CTA.
- **Why recommend Shopify POS**: 3-col row of feature blurbs
  (Get a unified back office / Close more sales / Offer flexible
  checkouts).
- **Selling Shopify POS**: section with 2 video cards (1:13 and
  10:18 durations) — render as `<figure>` with play-triangle SVG +
  duration chip; href to `#`.
- **How Shopify POS comes together**: 4 numbered step cards.

### 8. Partner docs — `/docs/partner`
- **H1**: "Partner docs"
- Search input at top.
- Two-col layout: left = sticky TOC nav (Getting started, Build
  apps, Build themes, Earn revenue, App listings, Partner Directory,
  Compliance), right = doc body with 3 sample article cards.

### 9. Product docs — `/docs/product`
- **H1**: "Product docs"
- Same layout as partner docs but TOC labels = Admin API, Storefront
  API, Webhooks, Functions, Polaris (UI), Hydrogen, Liquid.

### 10. Support — `/support`
- **H1**: "Support"
- Two-card layout:
  - Card 1 (75% width): "Visit the community forum" + paragraph +
    green CTA "Visit the forum" (links to `/support/forum` stub).
  - Card 2 (right column): "Partner support" small caption + 3-line
    contact paragraph with "Partner Support" inline link
    (`/support/contact`).
- **Sub-route**: `/support/contact` — form (Subject, Category
  dropdown, Description textarea) POSTing to `/support/tickets`
  creating an audit row. Confirmation page after submit.

### 11. Payouts — `/payouts`
- **H1**: "Payouts"; right-aligned green **Export CSV** secondary
  CTA.
- **Pending card** (white panel):
  - h2 "Pending" + right-aligned link "View pending transactions".
  - Big USD figure (e.g. `$2,151.43 USD`) + sub-caption "Estimated
    amount excluding taxes".
  - Below figure: "Estimated payout date Jun 22, 2026 via bank
    account (***05)".
- **Payouts history list**: 12 rows. Each row: left = link "May 16,
  2026 – June 1, 2026 – Sales" (`/payouts/<id>`); center = small grey
  "Sent Jun 5, 2026"; right = `$923.25 USD` right-aligned.
- **Sub-route**: `/payouts/<id>` — detail page showing line items
  (Date | Description | Amount).

### 12. Team — `/team`
- **H1**: "Team"
- Yellow banner at top: "Update emergency contact information to
  receive important API alerts" + outline CTA "Add contact
  information" + dismiss ×.
- **Owners** section: left col = label + description; right col =
  card with **Active / Invited** tabs, list of owner rows (avatar +
  name + "Last login about 1 hour ago"). Footer: outline button
  "Invite owner".
- **Staff members** section: same layout, list of staff users.
- **Sub-route**: `/team/invite` — form (Email, Role dropdown:
  owner|staff_business|staff_dev|staff_marketing|staff_support|
  staff_finance) POSTing to `/team/invites`.

### 13. Settings — `/settings`
- **H1**: "Partner account settings"
- Yellow emergency-contact banner at top.
- **Sections** (each: left description col + right white-card form
  col)
  - **Personal profile information** — avatar + name + email + lang
    + "Last login ..." + outline **Edit personal profile** CTA.
  - **Account information** — Partner ID label + value; App Store
    Registration label + green pill "Registered".
  - **Business details** — read-only fields (business name, type,
    industry).
  - **Contact information** — form (Business name, Website, Business
    email, Support email, Phone, Address 1, Address 2, City, ZIP,
    State, Country) with Cancel / Save row at bottom.
  - **Emergency developer contact information** — form (Name, Email,
    Phone).
  - **Payouts** — display payout method (Bank account ending ***05)
    + link "Edit payout method".

---

## Authentication

Single user signed in by default; mirrors the post-login Partners
shell. Provide a `/login` page (email + password form) that issues a
signed cookie session. Default credentials seeded:
`barada@example.com` / `password`. Anonymous access redirects to
login except `/login`, `/__env__/*`, `/static/*`.

## DB schema sketch

- `partners` — single row, the org (id, name, email, business_name,
  website, support_email, phone, address fields, partner_id=1146365).
- `users` — owners + staff (id, email, name, role, last_login_at,
  status `active|invited`).
- `stores` — id, name, slug (`mm-jop5`), kind (`client_transfer` /
  `collaborator` / `active_development`), last_login_at, status.
- `payouts` — id, period_start, period_end, sent_at, amount_cents,
  currency, method ("bank ***05"), status (`paid`|`pending`).
- `payout_line_items` — payout_id, date, description, amount_cents.
- `leads` — id, product (`plus`|`pos`|`plus_b2b`), merchant_name,
  contact_email, status (`submitted`|`qualified`|`won`|`lost`),
  earnings_cents, submitted_at.
- `referrals` — id, merchant_name, plan, status, earnings_cents.
- `directory_listings` — id, business_name, plan, review_status.
- `tickets` — support tickets.
- `themes` — id, name, status (`draft`|`in_review`|`approved`).
- `catalogs` — id, name, products_count, status.
- `audit_log` — id, occurred_at, operation, target_type, target_id,
  payload_json. Source of truth for oracles.

## Oracles (initial set)

1. `t01_submit_plus_lead` — audit row `lead_submitted` w/
   product=plus + leads row populated with merchant_name + email.
2. `t02_invite_staff_member` — `staff_invited` audit row w/ email +
   role; users row with status=invited.
3. `t03_export_payouts_csv` — `payouts_export_requested` audit
   row.
4. `t04_create_support_ticket` — `support_ticket_created` audit row
   + tickets row w/ subject + category + non-empty description.
5. `t05_update_business_email` — `settings_updated` audit row w/
   field=business_email; partners row email matches.
6. `t06_view_payout_detail` — `payout_viewed` audit row; payout_id
   resolves to valid row.
7. `t07_dismiss_emergency_banner` — `banner_dismissed` audit row
   with banner=emergency_contact.
8. `t08_partner_directory_request_review` — `directory_review_requested`
   audit row w/ business_name resolving to a `directory_listings`
   row.
9. `t09_search_stores_filter` — `stores_filter_applied` audit row
   with non-empty search OR status filter.
10. `t10_submit_pos_referral` — `lead_submitted` audit row w/
    product=pos.

All oracles run via `/__env__/oracle?task_id=tNN` returning
`{passed, score, reasons[], diff}`.
