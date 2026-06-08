# `/` — Home / marketing

Source: anchored on `deploy/sim_envs/_capture_brief.md` (parent session
captured the live mercor.com on 2026-06-08 via Chrome MCP @ 1512×677).
Direct re-capture not run this turn — flagged ⏳ in FIDELITY.md where
the brief leaves ambiguity.

## Brand & typography

- Font family: `Inter, "Inter Fallback", system-ui, sans-serif`
  (system stack fallback at runtime; no Google Fonts).
- Body background: white `#FFFFFF`.
- Body color: pure black `#000000`.
- Accent / brand: indigo-600 `#4F46E5` (`rgb(79, 70, 229)`).
- Surface greys (Tailwind palette, exact hex):
    - gray-100 `#F3F4F6`
    - gray-200 `#E5E7EB`
    - gray-400 `#9CA3AF`
    - gray-500 `#6B7280`
    - gray-600 `#4B5563`
    - gray-800 `#1F2937`

## Layout

- Slim sticky header; nav width compact (~374px) left-aligned.
- Header items, in order: `APEX`, `APEX-Agents`, `APEX-SWE`,
  `Research`, `Enterprise`, `Experts`. (We add `Sign in` / `Sign up`
  on the right of the bar; the live site puts these elsewhere but the
  CUA needs an entry point.)
- Hero H1: `Shape the frontier of AI`.
- Stats strip just below hero with three pseudo-stats:
    - `Average pay $/hr`
    - `Roles created (k)`
    - `Daily payouts ($)`
- "Latest roles" section H2 (only H2 in fold).
- Role card shape:
    - Role title (e.g. `Internal Medicine Expert`)
    - Rate range (e.g. `$130-$180/hr`)
    - 3-letter avatar badge (deterministic initials, e.g. `LFA`,
      `JDT`, `GBR`) — gray-100 background, gray-800 text.
    - `N hired recently` microcopy.
    - `Apply` CTA button (indigo-600 background, white text).

## Interaction triggers

- `Apply` button on each card → `GET /apply/<role_id>`.
- Header nav links → page changes only (no in-page nav drawer).

## Mirror priorities

1. Tailwind grey + indigo palette must match exactly.
2. 3-letter initial avatar pattern is distinctive — replicate as
   deterministic SVG / div blocks.
3. Card shape `title / $X-$Y/hr / NNN hired recently / Apply` is the
   canonical role card across home + jobs + experts.
