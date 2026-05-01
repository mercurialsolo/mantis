# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Email the
maintainers at `security@<your-domain>` (replace before publishing) with:

- A description of the issue
- Steps to reproduce
- The version / commit SHA you observed it on
- Any proof-of-concept code or logs (please redact secrets)

We aim to acknowledge reports within 3 business days and to ship a fix or
mitigation guidance within 30 days for high-severity issues.

## Supported versions

Mantis is pre-1.0. Only the latest released minor version receives security
patches. Once we cut `1.0.0` we will publish a support matrix here.

## Operational guidance

Mantis drives a real browser, accepts plans from API callers, and renders
arbitrary third-party web content. When you self-host:

- **Rotate any pre-shared `MANTIS_API_TOKEN` and tenant keys** if a worker
  image, log bundle, or screenshot may have leaked.
- **Use the per-tenant URL allowlist** (`tenant_auth.py`) — without it a
  caller-supplied plan can navigate to any host the worker can reach,
  including cloud metadata endpoints. Block `169.254.169.254` and
  `metadata.google.internal` at the network layer.
- **Treat screenshots and screencasts as PII**. They may capture form
  contents, session cookies (in URL fragments), or third-party user data.
  Apply the same retention / access controls you use for application logs.
- **Treat web content as adversarial**. Page text fed back into an LLM
  prompt is a prompt-injection vector. Constrain the extraction schema and
  validate model output against it on the server side.
- **Run the browser worker as an unprivileged user** in a container or VM
  with no host filesystem mounts beyond the per-run scratch dir.

## Out of scope

- DoS via large plans (we cap step count, runtime, and cost — see
  `MANTIS_MAX_*` env vars). Reports of cap-bypass are in scope.
- Cost overrun on third-party providers (Anthropic, Baseten, Modal). The
  per-tenant cost cap is best-effort, not a billing guarantee.
