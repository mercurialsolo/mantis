"""mantis-auth — a simulated authentication environment for agent eval.

A minimal SaaS console behind a multi-method auth wall: password, OAuth
(Google / GitHub / Microsoft / Okta), email magic-link, email OTP, and
passkey. The login surface is the embeddable :mod:`app.authflow` package;
the rest of this package is the host env (product pages, mock inbox,
seed, oracles, harness routes).
"""
