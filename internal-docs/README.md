# Internal docs — client-specific reference material

This directory holds docs that mention specific clients, internal-only
file paths, internal SHA references, or otherwise leak the identity or
shape of any one customer's integration. **Nothing here ships in the
public mkdocs site at `mercurialsolo.github.io/mantis/`.**

## Why this directory exists

Public documentation must work for any client without revealing who else
is on the platform. Listing client A's tenant ID, integration playbook,
or internal patch sites in the public docs:

- Discloses customer relationships A may not want disclosed.
- Embeds A's product surface (file paths, line numbers, internal repo
  layout) in our public surface — both fragile and a small confidentiality
  leak.
- Tempts other clients to copy A's pattern verbatim instead of writing
  the integration that fits their own stack.

Generic shape goes in `docs/`. Client-named specifics go here.

## What lives here

- `integrations/<client>.md` — per-client integration narrative,
  patch sites, settings, file paths. Mirrors of these in
  `staffai_tools/<client>/`-style internal docs are also fine.
- `PROPOSAL.md` — internal architecture decision docs that name
  specific clients, internal cost lines, or internal migration
  timelines. Move to `docs/appendix/` only after generalising.

## What does NOT live here

- Generic library / runner / API references → `docs/`.
- Per-tenant secrets, credentials, or PII → use the secret store, not
  the repo. (This directory is git-tracked but should never contain
  passwords, keys, or live customer data.)

## Adding a new client integration doc

1. Write the generic shape first under `docs/integrations/` if any of
   what you're documenting is reusable. Reusable patterns belong in
   the public doc.
2. Put client-specific details (their file paths, their tenant id,
   their CRM URL, their internal SHAs) in
   `internal-docs/integrations/<their-name>.md`.
3. Cross-link from the internal doc out to the public generic doc;
   never the other way around.

## CI guard

`tests/test_docs_client_isolation.py` greps `docs/` for known client
tokens and fails the build if any leak in. Add tokens to its denylist
when onboarding a new named client.
