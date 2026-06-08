# `/login` — Auth

Simple centered card. Email + password fields, primary CTA.

## Layout

- White page bg, centered card max-width 420px.
- Card has 1px gray-200 border, 12px radius.
- H1: `Sign in to Mercor`
- Email input, Password input, `Sign in` button (indigo-600).
- Below: `Don't have an account? Sign up` link.

## Mirror priorities

- ENV_REQUIRE_AUTH=1 triggers 303→/login on protected paths
  (/apply/*, /dashboard, /profile).
- After success: 303→ next param (default `/dashboard`).
