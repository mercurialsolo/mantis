"""Form-target grounding (#406) — locate a labelled input / button /
dropdown on any page and return pixel coordinates.

Split out of :mod:`mantis_agent.extraction.extractor` because the
methods being moved here aren't extraction — they return coordinates
for *actions*, not structured data read off a screenshot. Living in
:class:`ClaudeExtractor` was an accident of the code growing one
Anthropic API client and adding more methods to it.

Why this isn't named ``grounding/``: the sibling module
:mod:`mantis_agent.grounding` already owns ``GroundingModel`` /
``ClaudeGrounding`` / ``LLMGrounding`` — the legacy *click* grounding
surface used by :class:`MicroPlanRunner` for "refine an approximate
click target". That module is widely imported; promoting it to a
package would have shadowed every ``from mantis_agent.grounding import
ClaudeGrounding`` call site. The form-target work fits adjacent —
same conceptual cluster, separate namespace.

Public surface:

- :class:`FormTargetProvider` (in :mod:`.base`) — the protocol both
  the form handler and tests depend on.
- :class:`ClaudeFormTargetProvider` (in :mod:`.claude`) — the default
  implementation, wraps an :class:`AnthropicToolUseClient`.

A Holo3-backed provider is added in :mod:`.holo3` so the form
handler can route around Anthropic overload without rewriting the
prompt for each backend.
"""

from .base import DropdownVerifyResult, FormTargetProvider, FormTargetResult
from .claude import ClaudeFormTargetProvider
from .holo3 import Holo3FormTargetProvider

__all__ = [
    "FormTargetProvider",
    "FormTargetResult",
    "DropdownVerifyResult",
    "ClaudeFormTargetProvider",
    "Holo3FormTargetProvider",
]
