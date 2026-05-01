# Use cases — what Mantis is good at

Mantis is a generic computer-use agent. Anything a human does with a
keyboard, a mouse, and a browser, Mantis can drive — provided the steps
are explainable in a structured plan. This page is a tour of the
patterns that show up most often in production.

For copy-paste plans you can adapt, jump to [Recipes](../integrations/recipes.md).

---

## Read flows — pull structured data out of a UI

Mantis reads what's on the screen and returns a JSON row per item. The
schema is plan-driven; the same runtime handles vehicle listings, job
postings, news articles, and real-estate inventory without a code change.

| Pattern | Real-world target | Output |
|---|---|---|
| **Marketplace listings** | Vehicle / boat / RV listings; consumer-goods classifieds | `{year, make, model, price, phone, seller, url}` per listing |
| **Job postings** | Greenhouse, Lever, Workday, Ashby, public ATS pages | `{title, team, location, department, url}` per role |
| **Real-estate** | Zillow, Redfin, Trulia, MLS-public listings | `{address, price, beds, baths, sqft, agent, url}` per home |
| **Product catalog** | Amazon, Shopify storefronts, Walmart, Etsy | `{name, price, rating, availability, brand, url}` per product |
| **News / content** | Newsroom indices, blog archives, RSS-less sites | `{headline, byline, published, summary, url}` per article |
| **Admin user lookup** | Internal admin consoles ("find user by email") | `{email, full_name, plan, signup_date, last_login_at, billing_status, url}` |
| **Compliance / audit screens** | Console event logs, settings pages | Snapshot or row-by-row export, often with a recorded screencast |

→ See [Recipes 1–4 + 9](../integrations/recipes.md) for working plans.

---

## Write flows — change state in a SaaS UI

Same agent, different verbs. Form-shaped plans (`fill_field` /
`select_option` / `submit`) drive logged-in workflows on systems that
have no reliable public API for the target field.

| Pattern | Real-world target | Action |
|---|---|---|
| **CRM record edit** | Salesforce / HubSpot / Zoho / Pipedrive / custom CRMs | Open lead → edit field (status / industry / owner) → Save |
| **Contact upsert** | Same; webhook-driven sync from your own DB | New Contact → fill name/email/phone/owner → Save |
| **Stage moves** | ATS pipelines, deal stages, ticket statuses | Open record → change stage dropdown → confirm |
| **Refund / chargeback** | Stripe / Shopify / Square admin | Search by order ID → open payment → Refund → confirm amount |
| **Inventory adjust** | Shopify / Woo / NetSuite | Open product → set stock → save |
| **Customer-support reply** | Zendesk / Intercom / Front | Open ticket → paste templated reply → assign macro → submit |
| **Settings / config** | OAuth apps, billing portals, IAM consoles | Toggle setting → confirm dialog → snapshot post-state |

→ See [Recipes 5, 9, 10, 11](../integrations/recipes.md).

---

## Authenticated multi-step workflows

The agent is plan-driven, so end-to-end flows that span login,
navigation, search, edit, and verify all live in one plan. Examples:

- **Sales operations** — log into CRM → pull all leads in stage X →
  for each, update owner and post a Slack note via webhook
- **Recruiting hygiene** — log into ATS → walk pipeline → close stale
  candidates, move qualified to next stage
- **Customer-success motion** — log into product admin → snapshot
  feature-usage page → email customer the export
- **Bookkeeping** — log into bank/payments dashboard → reconcile a
  date range against your accounting system

Each step is a `submit` / `fill_field` / `select_option` / `click` /
`extract_data`. Failures inside a step are isolated; a failed `submit`
halts the plan before downstream damage.

---

## Social / publishing

The same form-flow vocabulary drives any web-based composer. Logging in
through the UI sidesteps the requirement of platform API access for
small-volume use cases.

| Pattern | Use case |
|---|---|
| **LinkedIn post** | Weekly product update, hiring announcement, employee shout-out |
| **Reddit submission** | Subreddit digest, AMA invitation, release note |
| **Instagram feed post** | Scheduled photo + caption + hashtags |
| **Twitter / X reply** | Customer-support response, thread continuation |

→ See [Recipes 6, 7, 8](../integrations/recipes.md).

> **Heads-up.** Action recipes carry real-world consequences. Always
> include a `gate: true` extract step that verifies the action posted,
> the refund cleared, the lead saved. Without that gate a plan can
> "succeed" while the underlying action silently failed. See the safety
> note at the bottom of [Recipes](../integrations/recipes.md#picking-the-right-loop_count-and-max_cost).

---

## Desktop tasks (Xvfb, not just browser)

The agent's runtime is `xdotool` driving any X application. Browser is
the most common target, but the same pipeline drives:

- **File manager** — open a folder, drag-drop into archive, rename a batch
- **Terminal** — run a command, capture stdout, paste into another app
- **LibreOffice / Office 365** — apply a style, fix headings, regenerate a ToC
- **Image / PDF tools** — crop, annotate, export

These need a desktop environment in the runner (Xvfb + window manager).
The Baseten and Modal images both ship with Xvfb + Chrome + xdotool;
adding `libreoffice-core` to the image extends them.

---

## Adversarial / anti-bot targets

Mantis drives a real Chrome via xdotool — no Playwright fingerprints, no
WebDriver flag, real user-agent. Sites with bot detection (Cloudflare,
PerimeterX, DataDome) usually let it through, especially when paired with
a residential proxy (`PROXY_URL` env var). Examples that have worked:

- Listing sites with Cloudflare Bot Fight
- E-commerce checkouts behind reCAPTCHA invisible
- Banking dashboards with TLS fingerprinting

Captchas remain user-visible only — Mantis surfaces a `gate_failed`
result so you can hand off to a captcha-solver step or bail early.

---

## When NOT to use Mantis

The agent costs ~$0.50 per minute of GPU + Claude calls. If a job has a
clean public API or a stable Playwright path, use that instead. Mantis
shines when:

- The target has no API, or the API doesn't expose the field you need
- The UI changes shape often enough that XPath/CSS selectors keep breaking
- The flow spans multiple apps and you want one plan, not a chain of glue
- A human can describe the task in 5–10 sentences

If the workflow is "fetch a row from a database" or "POST to a known
endpoint", do that — don't pay GPU time to render a screen and read it back.

---

## Next

- [Recipes](../integrations/recipes.md) — copy-paste plans for the 11
  most common patterns
- [Concepts](concepts.md) — the runtime model: plans, sections, gates, loops
- [Plan formats](plan-formats.md) — every step type, every field
- [Quickstart](quickstart.md) — run your first plan in 5 minutes
