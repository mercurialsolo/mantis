# /login + /signup — Fiverr auth

## Layout
Centered single column, max-width 360px, padding-top 80px.

### /login

- Fiverr logo top, 32px tall
- H1 "Sign in to your account" 24px / 700, centered
- Form fields stacked, 14px / 400, gap 16px:
  - Email/username — 48px tall input, 1px `#74767e` border, rounded 4px
  - Password — same
- "Forgot password?" link 13px / `#1dbf73` right-aligned
- Submit button — full-width, 48px, green `#1dbf73`, white text 16px /
  600, label "Continue"
- "—— Or ——" divider 12px / `#74767e`
- "Don't have an account? Join here" link 14px / `#1dbf73`

### /signup
Same shape with H1 "Create a new account" and additional Username
field. Submit "Join".

## Interactions
- POST /login → set session cookie → 303 to `next` or `/`
- POST /signup → create user + session cookie → 303 to `/`
