# mantis-auth — simulated authentication environment

A minimal SaaS console ("Mantis Console") behind a multi-method auth
wall. The product behind the wall is intentionally tiny; the value of
this env is the **auth surface** and the eight graded ways through it.

## Methods & scenarios

| Task id | Method | What the agent does |
|---------|--------|---------------------|
| `T01_password_login` | password | fill email + password, submit |
| `T02_oauth_google` | OAuth / Google | account picker → consent → callback |
| `T03_oauth_github` | OAuth / GitHub | dark "Authorize" chrome |
| `T04_oauth_microsoft` | OAuth / Microsoft | segmented permissions screen |
| `T05_oauth_okta` | OAuth / Okta | tenant grant screen |
| `T06_magic_link_email` | email magic link | request → read `/inbox` → click link |
| `T07_email_otp` | email OTP | request → read 6-digit code in `/inbox` → enter |
| `T08_passkey` | passkey (WebAuthn) | select a registered passkey (simulated assertion) |

Each OAuth provider renders its **own** account-picker + consent screens
so the agent perceives a distinct IdP per scenario. The password page
ships three layouts (`centered`, `split`, `minimal`).

## Seeded accounts

| account | email | enrolled methods |
|---------|-------|------------------|
| `user_00001` (Ada) | `ada@mantis.example` / `hunter2` | **every method** (canonical target) |
| `user_00002` (Grace, admin) | `grace@mantis.example` / `compiler1` | password, google, microsoft, passkey |
| `user_00003` (Alan) | `alan@mantis.example` | okta only (SSO-only, no password) |
| `user_00004..6` | `<name>@mantis.example` | password only |

## Routes

```
/                          public landing
/login                     password + provider buttons + alt methods (?layout=)
/auth/oauth/<p>/authorize  OAuth account picker         (p ∈ google|github|microsoft|okta)
/auth/oauth/<p>/consent    consent screen
/auth/oauth/<p>/grant      mint code → redirect interstitial
/auth/oauth/<p>/callback   exchange code → session
/auth/magic                request magic link
/auth/magic/verify         consume token → session
/auth/otp                  request one-time code
/auth/otp/verify           verify code → session
/auth/passkey              passkey picker
/auth/passkey/assert       verify assertion → session
/inbox, /inbox/{id}        mock mailbox (open while signed out)
/console, /account         the product, behind the wall
/__env__/*                 harness surface (X-Env-Admin gated, except /health)
```

## Run it

```bash
docker build -t mantis/sim-env-mantis-auth:latest deploy/sim_envs/mantis_auth
docker run --rm -p 8008:8080 \
    -e SEED=42 -e FAKE_NOW=2026-01-15T09:00:00Z \
    -e ENV_ADMIN_TOKEN=$(python -c 'import secrets;print(secrets.token_urlsafe(32))') \
    mantis/sim-env-mantis-auth:latest
```

Env vars: `AUTH_LAYOUT` (centered|split|minimal), `ENV_REQUIRE_AUTH`
(default 1), `AUTH_SESSION_SECRET`, `AUTH_SESSION_TTL_S`, `SEED`,
`FAKE_NOW`, `DB_PATH`.

## Grading

`GET /__env__/oracle?task_id=<T..>` (with `X-Env-Admin`) returns
`{passed, score, reasons, diff}`. Oracles read only the `mutations`
audit log + DB state, so verdicts are deterministic. Each scenario needs
a terminal `login_succeeded` for `user_00001` tagged with the right
`via` (+ `provider`); the email/passkey scenarios additionally require a
consumed token / verified code / asserted passkey, so a shortcut session
fails the gate.

## The drop-in auth flow (`app/authflow`)

The login surface is a **portable, storage-free package** — mount it on
any FastAPI env to gain the full method matrix:

```python
from app.authflow import build_auth_router, AuthConfig, sessions

config = AuthConfig(
    templates=my_jinja_templates,   # must contain the auth/*.html templates
    backend=MyAuthBackend(),        # implements the AuthBackend protocol
    app_name="My Product",
    post_login_redirect="/dashboard",
    enabled_methods=("password", "oauth", "passkey"),  # opt in/out per env
    enabled_providers=("google", "okta"),
    default_layout="split",
)
app.include_router(build_auth_router(config))
```

`AuthBackend` (see `app/authflow/config.py`) is a small protocol over
your own user store — user/oauth/passkey lookups, email delivery + token
consumption, and two audit hooks. `app/backend.py` is a worked SQLite
implementation you can copy. Nothing in `authflow` imports this env's
`db`, so it carries no coupling. Verify session cookies in your
middleware with `authflow.sessions.session_user_id(request)`.

## Layout

```
app/
  authflow/        ← the embeddable, env-agnostic auth flow
    config.py      AuthConfig + AuthBackend protocol
    router.py      build_auth_router(config) — all login routes
    sessions.py    signed session cookies
    passwords.py   password hashing
    tokens.py      ephemeral OAuth codes + passkey challenges
    providers.py   OAuth provider chrome registry
  backend.py       this env's AuthBackend over SQLite
  db.py            schema + audit log
  seed.py          deterministic accounts
  main.py          app factory: gate + wire authflow + product routes
  routes/          console (product), inbox (mailbox), env_admin (harness)
  oracles/         per-scenario graders
  templates/, static/
```
