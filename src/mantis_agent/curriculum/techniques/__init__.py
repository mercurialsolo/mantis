"""Curriculum techniques.

Each module in this package defines one technique with the following
module-level constants:

    NAME      str            — short human-readable label
    TAGS      list[str]      — semantic tags (boost embedding match)
    TRIGGERS  list[str]      — regex patterns that strongly suggest
                                relevance (used as a tiebreaker / boost)
    ALWAYS    bool           — if True, always include this technique
                                regardless of relevance score
    CONTENT   str            — the actual technique text shown to the model

The loader (curriculum.__init__) auto-discovers every module in this
directory at import time. To add a new technique, drop a new .py file
here following the same shape — no other code changes needed.
"""
