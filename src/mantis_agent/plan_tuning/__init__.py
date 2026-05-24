"""Plan-tuning layer — per-domain knobs the decomposer applies after
step generation.

See :mod:`.profiles` for the :class:`DomainProfile` dataclass, the
registry of known domains, and the
:func:`apply_domain_profile` pass.
"""

from .profiles import (
    DOMAIN_PROFILES,
    DomainProfile,
    apply_domain_profile,
    resolve_domain_profile,
)

__all__ = [
    "DOMAIN_PROFILES",
    "DomainProfile",
    "apply_domain_profile",
    "resolve_domain_profile",
]
