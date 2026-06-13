"""OAuth identity-provider registry — chrome + branding only.

Each provider renders its *own* account-picker + consent screens so the
agent sees a visually distinct IdP per scenario (Google's white card,
GitHub's dark "Authorize" page, Microsoft's segmented sign-in, Okta's
tenant login). The actual account list comes from the embedding env's
seeded ``oauth_identities`` — this module only supplies the look.

A provider is identified by a short slug used in the route
(``/auth/oauth/<slug>/authorize``) and in the ``provider`` field of the
``login_succeeded`` audit row that oracles grade on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    slug: str
    display_name: str
    # Jinja template rendering this provider's authorize/consent chrome.
    template: str
    # Accent colour used by the shared idp.css via inline --idp-accent.
    accent: str
    # Short tagline shown under the app name on the consent card.
    scopes: tuple[str, ...]


PROVIDERS: dict[str, Provider] = {
    "google": Provider(
        slug="google",
        display_name="Google",
        template="oauth_google.html",
        accent="#1a73e8",
        scopes=(
            "See your primary Google Account email address",
            "See your personal info, including any personal info "
            "you've made publicly available",
        ),
    ),
    "github": Provider(
        slug="github",
        display_name="GitHub",
        template="oauth_github.html",
        accent="#2da44e",
        scopes=(
            "Read your profile data",
            "Read your email addresses",
        ),
    ),
    "microsoft": Provider(
        slug="microsoft",
        display_name="Microsoft",
        template="oauth_microsoft.html",
        accent="#0067b8",
        scopes=(
            "Sign you in and read your profile",
            "Read your email address",
        ),
    ),
    "okta": Provider(
        slug="okta",
        display_name="Okta",
        template="oauth_okta.html",
        accent="#1662dd",
        scopes=(
            "View your profile and email",
        ),
    ),
}


def get_provider(slug: str) -> Provider | None:
    return PROVIDERS.get((slug or "").strip().lower())
